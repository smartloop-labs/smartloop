//! Port of `smartloop/api/routers/project_io.py`.
//!
//! Project export/import router.

use std::io::Write;
use std::path::PathBuf;

use axum::body::{Body, Bytes};
use axum::extract::multipart::MultipartRejection;
use axum::extract::{DefaultBodyLimit, Multipart, Path};
use axum::http::header::{CONTENT_DISPOSITION, CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use tokio::io::AsyncReadExt;

use crate::api::model::ProjectResponse;
use crate::api::services::get_model_service;
use crate::api::services::project_io::{export_project, import_project, ProjectIoError};

use super::_errors::{parse_model, py_error_kind, read_form, run_blocking, HTTPException, PyErrorKind};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.project_io";

/// Chunk size used when streaming the export archive.
pub const EXPORT_CHUNK_SIZE: usize = 64 * 1024;

/// Routes under `/v1` (tag `project-io`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/projects/{project_id}/export", post(export_project_endpoint))
        .route("/v1/projects/import", post(import_project_endpoint))
        // Starlette reads uploads of any size.
        .layer(DefaultBodyLimit::disable())
}

/// Deletes the exported archive once the response is done with it — after the
/// last chunk, or when the client goes away mid-download (the Python
/// generator's `finally`).
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        // Already gone is fine.
        let _ = std::fs::remove_file(&self.0);
    }
}

/// `POST /v1/projects/{project_id}/export` — export a project as a downloadable
/// zip archive (`application/zip`, `Content-Disposition: attachment`).  The
/// temporary zip is deleted once streamed.  404 when missing, 400 on `ValueError`.
pub async fn export_project_endpoint(Path(project_id): Path<String>) -> Result<Response, HTTPException> {
    let result = run_blocking(move || export_project(&project_id)).await?;
    let zip_path = match result {
        Ok(path) => path,
        Err(ProjectIoError::FileNotFoundError(msg)) => return Err(HTTPException::new(404, msg)),
        Err(ProjectIoError::ValueError(msg)) => return Err(HTTPException::new(400, msg)),
        Err(e) => {
            let e = anyhow::Error::from(e);
            return Err(match py_error_kind(&e) {
                PyErrorKind::FileNotFound => HTTPException::new(404, e.to_string()),
                PyErrorKind::Value => HTTPException::new(400, e.to_string()),
                _ => {
                    tracing::error!(target: LOGGER, "Error exporting project: {:?}", e);
                    HTTPException::new(500, e.to_string())
                }
            });
        }
    };

    let filename = zip_path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let cleanup = RemoveOnDrop(zip_path.clone());
    let stream = async_stream::stream! {
        let _cleanup = cleanup;
        let mut file = match tokio::fs::File::open(&zip_path).await {
            Ok(file) => file,
            Err(e) => {
                yield Err(e);
                return;
            }
        };
        loop {
            let mut chunk = vec![0u8; EXPORT_CHUNK_SIZE];
            match file.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => {
                    chunk.truncate(n);
                    yield Ok(Bytes::from(chunk));
                }
                Err(e) => {
                    yield Err(e);
                    return;
                }
            }
        }
    };

    Ok((
        [
            (CONTENT_TYPE, "application/zip".to_string()),
            (CONTENT_DISPOSITION, format!("attachment; filename=\"{filename}\"")),
        ],
        Body::from_stream(stream),
    )
        .into_response())
}

/// `POST /v1/projects/import` — import a project from a zip archive (multipart
/// `file`, optional form `name` overriding the project name).
///
/// Answered by the supervisor — `/v1/projects/import` names a verb, not a
/// project, so it is never forwarded to a child (see the middleware's
/// collection actions) — and the workspace it imports into has to be the one
/// this process serves from, or the project exists on disk and nowhere in the
/// process that just created it.
pub async fn import_project_endpoint(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<ProjectResponse>, HTTPException> {
    let form = read_form(multipart).await?;
    form.require(&["file"])?;
    let file = form.get("file").cloned().unwrap_or_default();
    let name = form.get("name").map(|f| f.text());
    let model_service = get_model_service();

    let result = run_blocking(move || -> anyhow::Result<serde_json::Map<String, serde_json::Value>> {
        // `NamedTemporaryFile(suffix=".zip", delete=False)`, removed when this
        // scope ends whatever happens.
        let mut tmp = tempfile::Builder::new().suffix(".zip").tempfile()?;
        tmp.write_all(&file.data)?;
        tmp.flush()?;

        let workspace = model_service.live_workspace()?;
        let mut workspace = workspace.write();
        let imported = import_project(tmp.path(), name.as_deref(), Some(&mut workspace))?;
        Ok(imported)
    })
    .await?;

    match result {
        Ok(project) => parse_model(serde_json::Value::Object(project)).map(Json),
        Err(e) if py_error_kind(&e) == PyErrorKind::Value => Err(HTTPException::new(400, e.to_string())),
        Err(e) => {
            tracing::error!(target: LOGGER, "Error importing project: {:?}", e);
            Err(HTTPException::new(500, e.to_string()))
        }
    }
}
