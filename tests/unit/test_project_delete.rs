//! Port of `tests/unit/test_project_delete.py`.
//!
//! Deleting a project, in the process that owns the workspace record.
//!
//! Removing a project is a write to `config.json`: the list of projects lives
//! nowhere else.  A per-project agent child cannot make that write — a pinned
//! `Workspace::save()` writes one `project.json` and nothing else — and least
//! of all for the project it is pinned to, which is the one leaving the list.
//! So the endpoint is answered by the supervisor, and these are the guarantees
//! that gives.

use std::fs;
use std::path::Path;
use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::Value;
use serial_test::serial;

use smartloop::api::services::model_service::ModelService;
use smartloop::config::app_settings::AppSettings;
use smartloop::model::model_manager::Model;
use smartloop::workspace::context::{active_project_id, delegate_projects, pin_project, use_project};
use smartloop::workspace::workspace::{
    MCPServer, Project, SharedWorkspace, Workspace, read_project_state,
};

use crate::shared::workspace_support::{LogCapture, TempHome};

const WORKSPACE_LOGGER: &str = "smartloop.workspace.workspace";

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
                    mcp_servers: vec![MCPServer {
                        id: format!("{pid}-srv"),
                        name: format!("{pid} server"),
                        ..Default::default()
                    }],
                    ..Default::default()
                })
                .collect(),
        );
    for project in &ws.projects {
        fs::create_dir_all(project.documents_dir()).unwrap();
    }
    ws.save().unwrap();
    ws
}

/// The `service` fixture: a `ModelService` wired to a real workspace but no
/// model.
///
/// `ensure_model` is what would otherwise pull the weights in; with a model
/// pipeline for the default model already in place it has nothing to do, and
/// everything `delete_project` touches after it is the workspace and the
/// filesystem.
struct Fixture {
    service: Arc<ModelService>,
    workspace: SharedWorkspace,
    home: TempHome,
}

fn service() -> Fixture {
    let home = TempHome::new();
    let workspace: SharedWorkspace = Arc::new(RwLock::new(workspace(&home)));
    let made = ModelService::new_bare();
    *made.model.write() = Some(Arc::new(Model::new("m", Arc::clone(&workspace), None)));
    *made.model_name.write() = Some(AppSettings::new().default_model);
    Fixture { service: made, workspace, home }
}

fn config(home: &TempHome) -> Value {
    serde_json::from_str(&fs::read_to_string(home.path().join("config.json")).unwrap()).unwrap()
}

fn project_ids(home: &TempHome) -> Vec<String> {
    config(home)["workspace"]["projects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["id"].as_str().unwrap().to_string())
        .collect()
}

fn load() -> Workspace {
    Workspace::load().expect("a workspace on disk")
}

fn ids(ws: &Workspace) -> Vec<String> {
    ws.projects.iter().map(|p| p.id.clone()).collect()
}

fn server_names(home: &TempHome, project_id: &str) -> Vec<String> {
    let state = read_project_state(&join(&home.join("ws-1"), project_id)).expect("a project state");
    state["mcp_servers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s["name"].as_str().unwrap().to_string())
        .collect()
}

fn not_in_workspace_warnings(caplog: &LogCapture) -> usize {
    caplog
        .warnings_from(WORKSPACE_LOGGER)
        .iter()
        .filter(|r| r.get_message().contains("is not in this workspace"))
        .count()
}

mod the_record_and_the_directory_go_together {
    use super::*;

    #[test]
    #[serial]
    fn test_the_project_leaves_the_workspace_record() {
        let fx = service();
        fx.service.delete_project("beta").unwrap();

        assert_eq!(project_ids(&fx.home), vec!["alpha"]);
        assert_eq!(ids(&load()), vec!["alpha"]);
    }

    #[test]
    #[serial]
    fn test_the_directory_is_removed() {
        let fx = service();
        fx.service.delete_project("beta").unwrap();

        assert!(!Path::new(&join(&fx.home.join("ws-1"), "beta")).exists());
    }

    /// The regression: the tree went, the record stayed, the UI relisted it.
    #[test]
    #[serial]
    fn test_it_does_not_come_back_on_the_next_load() {
        let fx = service();
        fx.service.delete_project("beta").unwrap();
        Workspace::reset();

        assert_eq!(ids(&load()), vec!["alpha"]);
    }

    #[test]
    #[serial]
    fn test_the_other_project_is_untouched() {
        let fx = service();
        fx.service.delete_project("beta").unwrap();

        assert_eq!(server_names(&fx.home, "alpha"), vec!["alpha server"]);
    }

    #[test]
    #[serial]
    fn test_deleting_the_selected_project_hands_the_selection_on() {
        let fx = service();
        fx.service.delete_project("alpha").unwrap();

        assert_eq!(load().current_project().unwrap().id, "beta");
    }

    /// Anything resolving the workspace afresh must not still see it.
    #[test]
    #[serial]
    fn test_the_singleton_is_dropped() {
        let fx = service();
        fx.service.delete_project("beta").unwrap();

        assert!(Workspace::get_instance().is_none());
    }

    #[test]
    #[serial]
    fn test_a_system_project_is_refused() {
        let fx = service();
        fx.workspace.write().projects[1].system = true;

        let err = fx.service.delete_project("beta").expect_err("refused");
        assert!(err.to_string().contains("System projects"), "unexpected error: {err}");

        assert!(Path::new(&join(&fx.home.join("ws-1"), "beta")).exists());
    }

    #[test]
    #[serial]
    fn test_an_unknown_project_is_refused() {
        let fx = service();
        let err = fx.service.delete_project("gamma").expect_err("refused");
        assert!(err.to_string().contains("Project not found"), "unexpected error: {err}");
    }
}

/// A delete arrives bound to the project it is about to remove.
///
/// The project-scope middleware binds the id out of the path, and that binding
/// is still live once the project is gone.  Resolving it then warns about an
/// absence this request was asked to create — and `save()` does resolve it.
/// The sentence it logs is the one a genuinely stale binding produces, so
/// emitting it here buries the real thing in false alarms.
mod the_binding_the_path_left_behind {
    use super::*;

    #[test]
    #[serial]
    fn test_a_delete_bound_to_its_own_project_is_quiet() {
        let fx = service();
        let caplog = LogCapture::start();

        {
            let _bound = use_project(Some("alpha"));
            fx.service.delete_project("alpha").unwrap();
        }

        assert_eq!(
            not_in_workspace_warnings(&caplog),
            0,
            "the delete warned about the project it had just removed"
        );
    }

    /// Unbinding must not change what the delete persists.
    #[test]
    #[serial]
    fn test_the_record_is_still_written_correctly() {
        let fx = service();
        {
            let _bound = use_project(Some("alpha"));
            fx.service.delete_project("alpha").unwrap();
        }

        assert_eq!(project_ids(&fx.home), vec!["beta"]);
        assert_eq!(config(&fx.home)["workspace"]["projects"][0]["current"], true);
    }

    /// Scoped, not cleared: whatever ran the delete gets its context back.
    #[test]
    #[serial]
    fn test_the_binding_is_restored_for_the_rest_of_the_request() {
        let fx = service();
        let _bound = use_project(Some("alpha"));
        fx.service.delete_project("alpha").unwrap();
        assert_eq!(active_project_id().as_deref(), Some("alpha"));
    }

    /// The warning has to survive for the case it was written for.
    ///
    /// Bound to a project this workspace never had — the symptom of a detached
    /// instance — it must still say so.
    #[test]
    #[serial]
    fn test_a_stale_binding_still_warns() {
        let fx = service();
        let caplog = LogCapture::start();

        {
            let _bound = use_project(Some("never-existed"));
            fx.workspace.read().save().unwrap();
        }

        assert!(not_in_workspace_warnings(&caplog) > 0);
    }
}

/// The supervisor has delegated the project files; it still owns this one.
mod under_supervision {
    use super::*;

    #[test]
    #[serial]
    fn test_the_delete_persists_while_projects_are_delegated() {
        let fx = service();
        delegate_projects(true);

        fx.service.delete_project("beta").unwrap();

        assert_eq!(project_ids(&fx.home), vec!["alpha"]);
        assert!(!Path::new(&join(&fx.home.join("ws-1"), "beta")).exists());
    }

    /// Deleting one project must not write another one's file back.
    #[test]
    #[serial]
    fn test_it_does_not_republish_a_delegated_project() {
        let fx = service();
        delegate_projects(true);
        // A child adds a server after the service loaded its workspace.
        let mut child = load();
        pin_project(Some("alpha"));
        child.projects[0].mcp_servers.push(MCPServer {
            id: "new".to_string(),
            name: "from the child".to_string(),
            ..Default::default()
        });
        child.save().unwrap();
        pin_project(None);

        fx.service.delete_project("beta").unwrap();

        assert_eq!(server_names(&fx.home, "alpha"), vec!["alpha server", "from the child"]);
    }
}

mod finding_the_project {
    use super::*;

    /// The cached workspace can predate the CLI, or an import.
    #[test]
    #[serial]
    fn test_a_project_created_by_another_process_can_be_deleted() {
        let fx = service();
        let mut on_disk = load();
        let ws_dir = fx.home.join("ws-1");
        let gamma = Project {
            id: "gamma".to_string(),
            project_dir: join(&ws_dir, "gamma"),
            current: false,
            name: Some("Gamma".to_string()),
            ..Default::default()
        };
        fs::create_dir_all(gamma.documents_dir()).unwrap();
        on_disk.projects.push(gamma);
        on_disk.save().unwrap();

        fx.service.delete_project("gamma").unwrap();

        assert_eq!(project_ids(&fx.home), vec!["alpha", "beta"]);
        assert!(!Path::new(&join(&ws_dir, "gamma")).exists());
    }
}
