//! A project remembers the client that made it (`client`), so a client can
//! list only its own kind: docent makes projects out of folders of code that
//! Studio has no use for.
use serde_json::json;
use smartloop::api::model::{project::CreateProjectRequest, project::ProjectResponse};
use smartloop::workspace::workspace::{MergeById, Project};

#[test]
fn a_project_carries_the_client_that_made_it_to_the_api() {
    let project = Project {
        id: "p1".into(),
        name: Some("todo-app".into()),
        client: Some("docent".into()),
        ..Project::default()
    };
    let value = serde_json::to_value(&project).unwrap();
    assert_eq!(value["client"], json!("docent"));
    let response: ProjectResponse = serde_json::from_value(value).unwrap();
    assert_eq!(response.client.as_deref(), Some("docent"));
}

#[test]
fn a_project_made_without_a_client_belongs_to_none() {
    // Configs written before the field existed, and clients that don't say.
    let project: Project = serde_json::from_value(json!({
        "id": "p2", "project_dir": "/tmp/p2", "current": false, "name": "notes"
    }))
    .unwrap();
    assert_eq!(project.client, None);
    let request: CreateProjectRequest = serde_json::from_value(json!({"name": "notes"})).unwrap();
    assert_eq!(request.client, None);
    let request: CreateProjectRequest =
        serde_json::from_value(json!({"name": "todo-app", "client": "docent"})).unwrap();
    assert_eq!(request.client.as_deref(), Some("docent"));
}

#[test]
fn carrying_a_project_keeps_its_client() {
    let mut target = Project { id: "p1".into(), ..Project::default() };
    let source = Project { id: "p1".into(), client: Some("docent".into()), ..Project::default() };
    target.carry_fields(source);
    assert_eq!(target.client.as_deref(), Some("docent"));
}

mod through_the_service {
    //! The API reads projects through `project_to_dict`, so the client has
    //! to come out there: set at creation, or claimed later by an update.
    use serial_test::serial;
    use smartloop::workspace::Workspace;
    use smartloop::workspace::context::{delegate_projects, pin_project};
    use crate::shared::model_service_b::{global_lock, service_with_workspace};

    #[test]
    #[serial]
    fn the_client_is_reported_and_can_be_claimed_later() {
        let _guard = global_lock();
        let tmp = tempfile::tempdir().expect("tempdir");
        let workspace_dir = tmp.path().to_string_lossy().into_owned();
        Workspace::set_config_file(Some(tmp.path().join("config.json").to_string_lossy().into_owned()));
        let svc = service_with_workspace(&workspace_dir, vec![], vec![]);

        let made = svc.create_project("todo-app", None, false, None, None, Some("docent")).expect("created");
        assert_eq!(made["client"], "docent");
        let old = svc.create_project("notes", None, false, None, None, None).expect("created");
        assert!(old["client"].is_null(), "a project made without a client has none");

        // Docent, run again in the folder, claims the project it made before
        // the field existed; a claim with "" lets it go.
        let id = old["id"].as_str().unwrap().to_string();
        let claimed = svc.update_project(&id, None, None, None, None, None, Some("docent")).expect("updated");
        assert_eq!(claimed["client"], "docent");
        {
            // What the workspace now holds, which every later listing reads.
            let model = svc.model.read().clone().expect("model");
            let workspace = model.workspace.read();
            assert!(workspace.projects.iter().any(|p| p.id == id && p.client.as_deref() == Some("docent")));
        }
        let released = svc.update_project(&id, None, None, None, None, None, Some("")).expect("updated");
        assert!(released["client"].is_null());
        // An update that says nothing about the client leaves it alone.
        let kept = svc.update_project(&made["id"].as_str().unwrap().to_string(), Some("todo"), None, None, None, None, None).expect("updated");
        assert_eq!(kept["client"], "docent");

        pin_project(None);
        delegate_projects(false);
        Workspace::set_config_file(None);
    }
}
