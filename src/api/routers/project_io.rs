//! Port of `smartloop/api/routers/project_io.py`.
//!
//! Project export/import router.

use axum::extract::{Multipart, Path};
use axum::response::Response;
use axum::routing::post;
use axum::{Json, Router};

use crate::api::model::ProjectResponse;
#[allow(unused_imports)]
use crate::api::services::project_io::{export_project, import_project};
#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.project_io";

/// Chunk size used when streaming the export archive.
pub const EXPORT_CHUNK_SIZE: usize = 64 * 1024;

/// Routes under `/v1` (tag `project-io`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/projects/{project_id}/export", post(export_project_endpoint))
        .route("/v1/projects/import", post(import_project_endpoint))
}

/// `POST /v1/projects/{project_id}/export` — export a project as a downloadable
/// zip archive (`application/zip`, `Content-Disposition: attachment`).  The
/// temporary zip is deleted once streamed.  404 when missing, 400 on `ValueError`.
pub async fn export_project_endpoint(Path(project_id): Path<String>) -> Result<Response, HTTPException> {
    let _ = project_id;
    todo!()
}

/// `POST /v1/projects/import` — import a project from a zip archive (multipart
/// `file`, optional form `name` overriding the project name).
///
/// Answered by the supervisor — `/v1/projects/import` names a verb, not a
/// project, so it is never forwarded to a child — and the workspace it imports
/// into has to be the one this process serves from.
pub async fn import_project_endpoint(multipart: Multipart) -> Result<Json<ProjectResponse>, HTTPException> {
    let _ = multipart;
    todo!()
}
