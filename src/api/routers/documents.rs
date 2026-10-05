//! Port of `smartloop/api/routers/documents.py`.
//!
//! Document management router.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Multipart, Path};
use axum::response::Response;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::Stream;
use serde_json::Value;

use crate::api::model::{AddDocumentRequest, DeleteDocumentResponse, DocumentListResponse};
#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};
#[allow(unused_imports)]
use crate::workspace::{DocumentInfo, DocumentStatus};

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.documents";

/// Statuses after which a document is no longer processed.
pub const TERMINAL_STATUSES: [&str; 4] = ["indexed", "failed", "deleted", "processed"];
/// Time between status checks.
pub const POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Max time before the progress stream closes.
pub const STREAM_TIMEOUT: Duration = Duration::from_secs(300);

/// Routes under `/v1` (tag `documents`).
pub fn router() -> Router {
    Router::new()
        .route(
            "/v1/projects/{project_id}/documents",
            get(list_documents_for_project).post(add_document_to_project),
        )
        .route("/v1/projects/{project_id}/documents/upload", post(upload_document_file))
        .route(
            "/v1/projects/{project_id}/documents/{document_id}",
            delete(delete_document_from_project),
        )
}

/// Format a JSON value as an SSE event string
/// (`event: <event>\ndata: <json.dumps(data)>\n\n`).
pub fn sse_event(data: &Value, event: &str) -> String {
    let _ = (data, event);
    todo!()
}

/// Yield SSE events (`queued`, `progress`, `done`, or `timeout`) until all
/// documents reach a terminal status.
pub fn stream_document_progress(
    doc_ids: Vec<String>,
    model_service: Arc<ModelService>,
    project_id: Option<String>,
) -> impl Stream<Item = String> + Send + 'static {
    let _ = (doc_ids, model_service, project_id);
    futures::stream::empty::<String>()
}

/// Yield a single SSE `done` event for requests that need no processing.
pub fn sse_done_immediately(message: String) -> impl Stream<Item = String> + Send + 'static {
    let _ = message;
    futures::stream::empty::<String>()
}

/// `GET /v1/projects/{project_id}/documents` — list documents for a project
/// (404 on `ValueError`, 503 on `RuntimeError`).
pub async fn list_documents_for_project(
    Path(project_id): Path<String>,
) -> Result<Json<DocumentListResponse>, HTTPException> {
    let _ = project_id;
    todo!()
}

/// `POST /v1/projects/{project_id}/documents` — add a document source and
/// stream progress as `text/event-stream` (headers `Cache-Control: no-cache`,
/// `X-Accel-Buffering: no`).
pub async fn add_document_to_project(
    Path(project_id): Path<String>,
    Json(request): Json<AddDocumentRequest>,
) -> Result<Response, HTTPException> {
    let _ = (project_id, request);
    todo!()
}

/// `POST /v1/projects/{project_id}/documents/upload` — upload a file (multipart
/// `file` + form `cwd`) into `<cwd>/documents/`; the indexer picks it up.
/// Returns immediately with `status=accepted`.
pub async fn upload_document_file(
    Path(project_id): Path<String>,
    multipart: Multipart,
) -> Result<Json<Value>, HTTPException> {
    let _ = (project_id, multipart);
    todo!()
}

/// `DELETE /v1/projects/{project_id}/documents/{document_id}` — delete a document.
pub async fn delete_document_from_project(
    Path((project_id, document_id)): Path<(String, String)>,
) -> Result<Json<DeleteDocumentResponse>, HTTPException> {
    let _ = (project_id, document_id);
    todo!()
}
