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
