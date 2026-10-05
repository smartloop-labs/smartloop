//! Port of `smartloop/api/routers/projects.py`.
//!
//! Project CRUD router.

use axum::extract::Path;
use axum::routing::{patch, post};
use axum::{Json, Router};

use crate::api::model::{
    CreateProjectRequest, DeleteProjectResponse, ProjectListResponse, ProjectResponse,
    UpdateProjectRequest,
};
use crate::api::model::project::SkillRequest;
#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};
#[allow(unused_imports)]
use crate::supervisor::service::release_project;
use crate::workspace::workspace::Skill;

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.projects";

/// Routes under `/v1` (tag `projects`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/projects", post(create_project).get(list_projects))
        .route("/v1/projects/{project_id}", patch(update_project).delete(delete_project))
}

/// Build a workspace [`Skill`] from a request: the id is the md5 hex of the
/// content, and a missing name becomes `skill-<first 6 hex chars>`.
pub fn skill_from_request(request: &SkillRequest) -> Skill {
    let _ = request;
    todo!()
}

/// `POST /v1/projects` — create a new project (400 on `ValueError`, 503 on `RuntimeError`).
pub async fn create_project(
    Json(request): Json<CreateProjectRequest>,
) -> Result<Json<ProjectResponse>, HTTPException> {
    let _ = request;
    todo!()
}

/// `GET /v1/projects` — list all projects.
pub async fn list_projects() -> Result<Json<ProjectListResponse>, HTTPException> {
    todo!()
}

/// `PATCH /v1/projects/{project_id}` — update a project (404 on `ValueError`).
pub async fn update_project(
    Path(project_id): Path<String>,
    Json(request): Json<UpdateProjectRequest>,
) -> Result<Json<ProjectResponse>, HTTPException> {
    let _ = (project_id, request);
    todo!()
}

/// `DELETE /v1/projects/{project_id}` — delete a project by ID.
///
/// Answered here rather than by the project's own agent: the project record
/// lives in the workspace file, which only this process writes.  So the agent
/// has to be stopped from here too — and before the record goes, because the
/// pool finds the child's port files through the workspace entry it is about
/// to lose.
pub async fn delete_project(
    Path(project_id): Path<String>,
) -> Result<Json<DeleteProjectResponse>, HTTPException> {
    let _ = project_id;
    todo!()
}
