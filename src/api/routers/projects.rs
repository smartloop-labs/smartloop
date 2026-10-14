//! Port of `smartloop/api/routers/projects.py`.
//!
//! Project CRUD router.

use axum::extract::Path;
use axum::routing::{patch, post};
use axum::{Json, Router};
use md5::{Digest, Md5};

use crate::api::model::project::SkillRequest;
use crate::api::model::{
    CreateProjectRequest, DeleteProjectResponse, ProjectListResponse, ProjectResponse,
    UpdateProjectRequest,
};
use crate::api::services::get_model_service;
use crate::supervisor::service::release_project;
use crate::workspace::workspace::Skill;

use super::_errors::{parse_model, py_error_kind, run_blocking, HTTPException, PyErrorKind, ValidatedJson};

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
    let id = hex::encode(Md5::digest(request.content.as_bytes()));
    let name = match request.name.as_deref() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => format!("skill-{}", &id[..6]),
    };
    Skill {
        id,
        name,
        description: request.description.clone(),
        content: request.content.clone(),
        system: false,
    }
}

/// Map a service failure: `ValueError` → *value_status*, `RuntimeError` →
/// 503, anything else → 500 (logged as *what*).
fn project_error(err: anyhow::Error, value_status: u16, what: &str) -> HTTPException {
    match py_error_kind(&err) {
        PyErrorKind::Value => HTTPException::new(value_status, err.to_string()),
        PyErrorKind::Runtime => HTTPException::new(503, err.to_string()),
        _ => {
            tracing::error!(target: LOGGER, "{}: {:?}", what, err);
            HTTPException::new(500, err.to_string())
        }
    }
}

/// `POST /v1/projects` — create a new project (400 on `ValueError`, 503 on `RuntimeError`).
#[utoipa::path(
    post,
    path = "/v1/projects",
    tag = "projects",
    request_body = CreateProjectRequest,
    summary = "Create a project",
    description = "",
    responses((status = 200, description = "OK", body = ProjectResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn create_project(
    ValidatedJson(request): ValidatedJson<CreateProjectRequest>,
) -> Result<Json<ProjectResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        let skills: Vec<Skill> = request.skills.iter().map(skill_from_request).collect();
        model_service.create_project(
            &request.name,
            request.description.as_deref(),
            request.system,
            Some(skills),
            request.cwd.as_deref(),
        )
    })
    .await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(project_error(e, 400, "Error creating project")),
    }
}

/// `GET /v1/projects` — list all projects.
#[utoipa::path(
    get,
    path = "/v1/projects",
    tag = "projects",
    summary = "List projects",
    description = "",
    responses((status = 200, description = "OK", body = ProjectListResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_projects() -> Result<Json<ProjectListResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || model_service.list_projects()).await?;
    match result {
        Ok(projects) => {
            let projects = projects
                .into_iter()
                .map(parse_model::<ProjectResponse>)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Json(ProjectListResponse { projects }))
        }
        // Only a RuntimeError is singled out here; a ValueError is a 500.
        Err(e) if py_error_kind(&e) == PyErrorKind::Runtime => Err(HTTPException::new(503, e.to_string())),
        Err(e) => {
            tracing::error!(target: LOGGER, "Error listing projects: {:?}", e);
            Err(HTTPException::new(500, e.to_string()))
        }
    }
}

/// `PATCH /v1/projects/{project_id}` — update a project (404 on `ValueError`).
#[utoipa::path(
    patch,
    path = "/v1/projects/{project_id}",
    tag = "projects",
    params(("project_id" = String, Path, description = "Project ID")),
    request_body = UpdateProjectRequest,
    summary = "Update a project",
    description = "",
    responses((status = 200, description = "OK", body = ProjectResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn update_project(
    Path(project_id): Path<String>,
    ValidatedJson(request): ValidatedJson<UpdateProjectRequest>,
) -> Result<Json<ProjectResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        let skills: Vec<Skill> = request.skills.iter().map(skill_from_request).collect();
        model_service.update_project(
            &project_id,
            request.name.as_deref(),
            request.description.as_deref(),
            Some(skills),
            request.web_search_enabled,
        )
    })
    .await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(project_error(e, 404, "Error updating project")),
    }
}

/// `DELETE /v1/projects/{project_id}` — delete a project by ID.
///
/// Answered here rather than by the project's own agent: the project record
/// lives in the workspace file, which only this process writes (see
/// `crate::supervisor::routing`).  So the agent has to be stopped from here
/// too — and before the record goes, because the pool finds the child's port
/// files through the workspace entry it is about to lose.
#[utoipa::path(
    delete,
    path = "/v1/projects/{project_id}",
    tag = "projects",
    params(("project_id" = String, Path, description = "Project ID")),
    summary = "Delete a project",
    description = "",
    responses((status = 200, description = "OK", body = DeleteProjectResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn delete_project(
    Path(project_id): Path<String>,
) -> Result<Json<DeleteProjectResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        if release_project(&project_id) {
            tracing::info!(target: LOGGER, "[projects] stopped the agent for {} before deleting it", project_id);
        }
        model_service.delete_project(&project_id)
    })
    .await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(project_error(e, 404, "Error deleting project")),
    }
}
