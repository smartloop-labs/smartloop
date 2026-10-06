//! Port of `smartloop/api/routers/skills.py`.
//!
//! Skills CRUD router.

use axum::extract::Path;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::api::model::project::{SkillRequest, SkillResponse, UpdateSkillRequest};
use crate::api::services::get_model_service;

use super::_errors::{parse_model, py_error_kind, run_blocking, HTTPException, PyErrorKind, ValidatedJson};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.skills";

/// Routes under `/v1` (tag `skills`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/projects/{project_id}/skills", get(list_skills).post(add_skill))
        .route(
            "/v1/projects/{project_id}/skills/{skill_id}",
            get(get_skill).patch(update_skill).delete(delete_skill),
        )
}

/// `ValueError` → 404, anything else → 500 (logged as *what*).
fn skill_error(err: anyhow::Error, what: &str) -> HTTPException {
    if py_error_kind(&err) == PyErrorKind::Value {
        return HTTPException::new(404, err.to_string());
    }
    tracing::error!(target: LOGGER, "{}: {:?}", what, err);
    HTTPException::new(500, err.to_string())
}

/// `GET /v1/projects/{project_id}/skills` — `{"skills": [SkillResponse...]}`.
pub async fn list_skills(Path(project_id): Path<String>) -> Result<Json<Value>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || model_service.list_skills(&project_id)).await?;
    match result {
        Ok(skills) => {
            let skills = skills
                .into_iter()
                .map(parse_model::<SkillResponse>)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Json(json!({ "skills": skills })))
        }
        Err(e) => Err(skill_error(e, "Error listing skills")),
    }
}

/// `POST /v1/projects/{project_id}/skills` — add a skill to a project.
pub async fn add_skill(
    Path(project_id): Path<String>,
    ValidatedJson(request): ValidatedJson<SkillRequest>,
) -> Result<Json<SkillResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        model_service.add_skill(
            &project_id,
            &request.content,
            request.name.as_deref(),
            request.description.as_deref(),
        )
    })
    .await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(skill_error(e, "Error adding skill")),
    }
}

/// `GET /v1/projects/{project_id}/skills/{skill_id}` — get a single skill by ID.
pub async fn get_skill(
    Path((project_id, skill_id)): Path<(String, String)>,
) -> Result<Json<SkillResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || model_service.get_skill(&project_id, &skill_id)).await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(skill_error(e, "Error getting skill")),
    }
}

/// `PATCH /v1/projects/{project_id}/skills/{skill_id}` — update a skill's content.
pub async fn update_skill(
    Path((project_id, skill_id)): Path<(String, String)>,
    ValidatedJson(request): ValidatedJson<UpdateSkillRequest>,
) -> Result<Json<SkillResponse>, HTTPException> {
    let model_service = get_model_service();
    let result =
        run_blocking(move || model_service.update_skill(&project_id, &skill_id, &request.content)).await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(skill_error(e, "Error updating skill")),
    }
}

/// `DELETE /v1/projects/{project_id}/skills/{skill_id}` — delete a skill by ID.
pub async fn delete_skill(
    Path((project_id, skill_id)): Path<(String, String)>,
) -> Result<Json<Value>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || model_service.delete_skill(&project_id, &skill_id)).await?;
    match result {
        Ok(value) => Ok(Json(value)),
        Err(e) => Err(skill_error(e, "Error deleting skill")),
    }
}
