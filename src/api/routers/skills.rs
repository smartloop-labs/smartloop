//! Port of `smartloop/api/routers/skills.py`.
//!
//! Skills CRUD router.

use axum::extract::Path;
use axum::routing::get;
use axum::{Json, Router};
use serde_json::Value;

use crate::api::model::project::{SkillRequest, SkillResponse, UpdateSkillRequest};
#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};

use super::_errors::HTTPException;

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

/// `GET /v1/projects/{project_id}/skills` — `{"skills": [SkillResponse...]}`.
pub async fn list_skills(Path(project_id): Path<String>) -> Result<Json<Value>, HTTPException> {
    let _ = project_id;
    todo!()
}

/// `POST /v1/projects/{project_id}/skills` — add a skill to a project.
pub async fn add_skill(
    Path(project_id): Path<String>,
    Json(request): Json<SkillRequest>,
) -> Result<Json<SkillResponse>, HTTPException> {
    let _ = (project_id, request);
    todo!()
}

/// `GET /v1/projects/{project_id}/skills/{skill_id}` — get a single skill by ID.
pub async fn get_skill(
    Path((project_id, skill_id)): Path<(String, String)>,
) -> Result<Json<SkillResponse>, HTTPException> {
    let _ = (project_id, skill_id);
    todo!()
}

/// `PATCH /v1/projects/{project_id}/skills/{skill_id}` — update a skill's content.
pub async fn update_skill(
    Path((project_id, skill_id)): Path<(String, String)>,
    Json(request): Json<UpdateSkillRequest>,
) -> Result<Json<SkillResponse>, HTTPException> {
    let _ = (project_id, skill_id, request);
    todo!()
}

/// `DELETE /v1/projects/{project_id}/skills/{skill_id}` — delete a skill by ID.
pub async fn delete_skill(
    Path((project_id, skill_id)): Path<(String, String)>,
) -> Result<Json<Value>, HTTPException> {
    let _ = (project_id, skill_id);
    todo!()
}
