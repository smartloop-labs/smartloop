//! Port of `smartloop/api/routers/documents.py`.
//!
//! Document management router.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::multipart::MultipartRejection;
use axum::extract::{DefaultBodyLimit, Multipart, Path};
use axum::http::header::{CACHE_CONTROL, CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use futures::{Stream, StreamExt};
use once_cell::sync::Lazy;
use serde_json::{json, Value};
use tokio::sync::Semaphore;
use tokio::time::Instant;

use crate::api::model::{AddDocumentRequest, DeleteDocumentResponse, DocumentListResponse, DocumentResponse};
use crate::api::services::{get_model_service, ModelService};
use crate::workspace::{DocumentInfo, DocumentStatus};

use super::_errors::{
    json_dumps, parse_model, py_error_kind, read_form, run_on, HTTPException, PyErrorKind, ValidatedJson,
};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.documents";

/// Statuses after which a document is no longer processed.
pub const TERMINAL_STATUSES: [&str; 4] = ["indexed", "failed", "deleted", "processed"];
/// Time between status checks.
pub const POLL_INTERVAL: Duration = Duration::from_secs(1);
/// Max time before the progress stream closes.
pub const STREAM_TIMEOUT: Duration = Duration::from_secs(300);

/// The router's `ThreadPoolExecutor(max_workers=1)`: every blocking document
/// call — the routes and the progress poll alike — runs one at a time.
static EXECUTOR: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(1));

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
        // Starlette reads uploads of any size.
        .layer(DefaultBodyLimit::disable())
}

/// Format a JSON value as an SSE event string
/// (`event: <event>\ndata: <json.dumps(data)>\n\n`).
pub fn sse_event(data: &Value, event: &str) -> String {
    format!("event: {}\ndata: {}\n\n", event, json_dumps(data))
}

fn is_terminal(status: Option<&String>) -> bool {
    status.is_some_and(|s| TERMINAL_STATUSES.contains(&s.as_str()))
}

fn document_info_json(info: &DocumentInfo) -> Value {
    serde_json::to_value(info).unwrap_or(Value::Null)
}

/// Yield SSE events (`queued`, `progress`, `done`, or `timeout`) until all
/// documents reach a terminal status.
pub fn stream_document_progress(
    doc_ids: Vec<String>,
    model_service: Arc<ModelService>,
    project_id: Option<String>,
) -> impl Stream<Item = String> + Send + 'static {
    async_stream::stream! {
        let pending = DocumentStatus::Pending.as_str().to_string();
        let mut last_statuses: HashMap<Option<String>, String> =
            doc_ids.iter().map(|id| (Some(id.clone()), pending.clone())).collect();
        let deadline = Instant::now() + STREAM_TIMEOUT;

        // Emit an initial event so the client sees the queued documents immediately.
        yield sse_event(
            &json!({
                "documents": doc_ids.iter().map(|id| json!({"id": id, "status": pending})).collect::<Vec<_>>(),
                "message": format!("{} document(s) queued for processing", doc_ids.len()),
            }),
            "queued",
        );

        while Instant::now() < deadline {
            tokio::time::sleep(POLL_INTERVAL).await;

            let svc = model_service.clone();
            let ids = doc_ids.clone();
            let pid = project_id.clone();
            let polled = run_on(&EXECUTOR, move || svc.get_document_statuses(&ids, pid.as_deref())).await;
            let doc_infos: Vec<DocumentInfo> = match polled {
                Ok(Ok(infos)) => infos,
                Ok(Err(e)) => {
                    // The generator raising mid-stream: the response ends here.
                    tracing::error!(target: LOGGER, "Error polling document statuses: {:?}", e);
                    return;
                }
                Err(e) => {
                    tracing::error!(target: LOGGER, "Error polling document statuses: {}", e.detail);
                    return;
                }
            };

            let mut changed: Vec<Value> = Vec::new();
            for info in &doc_infos {
                if last_statuses.get(&info.id) != Some(&info.status) {
                    last_statuses.insert(info.id.clone(), info.status.clone());
                    changed.push(document_info_json(info));
                }
            }

            if !changed.is_empty() {
                yield sse_event(&json!({ "documents": changed }), "progress");
            }

            let all_done = doc_ids
                .iter()
                .all(|id| is_terminal(last_statuses.get(&Some(id.clone()))));
            if all_done {
                yield sse_event(
                    &json!({
                        "documents": doc_infos.iter().map(document_info_json).collect::<Vec<_>>(),
                        "message": "Processing complete",
                    }),
                    "done",
                );
                return;
            }
        }

        // Timeout reached — tell the client to poll on their own.
        yield sse_event(&json!({"message": "Stream timeout — poll /documents for status"}), "timeout");
    }
}

/// Yield a single SSE `done` event for requests that need no processing.
pub fn sse_done_immediately(message: String) -> impl Stream<Item = String> + Send + 'static {
    futures::stream::once(async move { sse_event(&json!({"documents": [], "message": message}), "done") })
}

/// A `text/event-stream` response with the headers the Python router set.
fn event_stream_response<S>(stream: S) -> Response
where
    S: Stream<Item = String> + Send + 'static,
{
    let body = Body::from_stream(stream.map(Ok::<_, Infallible>));
    (
        [
            (CONTENT_TYPE, "text/event-stream; charset=utf-8"),
            (CACHE_CONTROL, "no-cache"),
        ],
        [("X-Accel-Buffering", "no")],
        body,
    )
        .into_response()
}

/// `GET /v1/projects/{project_id}/documents` — list documents for a project
/// (404 on `ValueError`, 503 on `RuntimeError`).
#[utoipa::path(
    get,
    path = "/v1/projects/{project_id}/documents",
    tag = "documents",
    params(("project_id" = String, Path, description = "Project ID")),
    summary = "List documents",
    description = "",
    responses((status = 200, description = "OK", body = DocumentListResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_documents_for_project(
    Path(project_id): Path<String>,
) -> Result<Json<DocumentListResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_on(&EXECUTOR, move || model_service.get_documents_for_project(&project_id)).await?;
    match result {
        Ok(docs) => {
            let documents = docs
                .into_iter()
                .map(parse_model::<DocumentResponse>)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Json(DocumentListResponse { documents }))
        }
        Err(e) => match py_error_kind(&e) {
            PyErrorKind::Value => Err(HTTPException::new(404, e.to_string())),
            PyErrorKind::Runtime => Err(HTTPException::new(503, e.to_string())),
            _ => {
                tracing::error!(target: LOGGER, "Error listing documents for project: {:?}", e);
                Err(HTTPException::new(500, e.to_string()))
            }
        },
    }
}

/// The ids of the documents an `add_document_async` result queued
/// (`[d["id"] for d in result.get("documents", [])]`).
fn queued_ids(result: &Value) -> Result<Vec<String>, String> {
    let documents = match result.get("documents") {
        None => return Ok(Vec::new()),
        Some(Value::Array(documents)) => documents,
        Some(other) => return Err(format!("documents is not a list: {other}")),
    };
    documents
        .iter()
        .map(|d| match d.get("id") {
            Some(Value::String(id)) => Ok(id.clone()),
            Some(other) => Ok(other.to_string()),
            None => Err("'id'".to_string()),
        })
        .collect()
}

/// `POST /v1/projects/{project_id}/documents` — add a document source and
/// stream progress as `text/event-stream` (headers `Cache-Control: no-cache`,
/// `X-Accel-Buffering: no`).
///
/// Returns a text/event-stream response with the same events as POST /documents:
/// `queued`, `progress`…, then `done` (or `timeout`).
#[utoipa::path(
    post,
    path = "/v1/projects/{project_id}/documents",
    tag = "documents",
    params(("project_id" = String, Path, description = "Project ID")),
    request_body = AddDocumentRequest,
    summary = "Add a document source",
    description = "Indexes a file, directory or URL and streams progress as Server-Sent Events.",
    responses((status = 200, description = "Indexing progress as an SSE stream", content_type = "text/event-stream"), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn add_document_to_project(
    Path(project_id): Path<String>,
    ValidatedJson(request): ValidatedJson<AddDocumentRequest>,
) -> Result<Response, HTTPException> {
    let model_service = get_model_service();
    let svc = model_service.clone();
    let pid = project_id.clone();
    let result = run_on(&EXECUTOR, move || {
        svc.add_document_async(&request.source, Some(&pid), request.cwd.as_deref(), request.generate_filename)
    })
    .await?;

    let result = match result {
        Ok(result) => result,
        Err(e) if e.downcast_ref::<HTTPException>().is_some() => {
            return Err(e.downcast::<HTTPException>().unwrap_or_else(|e| HTTPException::new(500, e.to_string())));
        }
        Err(e) if py_error_kind(&e) == PyErrorKind::Runtime => {
            return Err(HTTPException::new(503, e.to_string()));
        }
        Err(e) => {
            tracing::error!(target: LOGGER, "Error adding document to project: {:?}", e);
            return Err(HTTPException::new(500, e.to_string()));
        }
    };

    let doc_ids = queued_ids(&result).map_err(|e| {
        tracing::error!(target: LOGGER, "Error adding document to project: {}", e);
        HTTPException::new(500, e)
    })?;
    if doc_ids.is_empty() {
        let message = match result.get("message") {
            Some(Value::String(message)) => message.clone(),
            Some(other) => other.to_string(),
            None => "Document(s) already exist".to_string(),
        };
        return Ok(event_stream_response(sse_done_immediately(message)));
    }

    Ok(event_stream_response(stream_document_progress(doc_ids, model_service, Some(project_id))))
}

/// `POST /v1/projects/{project_id}/documents/upload` — upload a file (multipart
/// `file` + form `cwd`) into `<cwd>/documents/`; the indexer picks it up
/// automatically within a few seconds.  Returns immediately with
/// `status=accepted`; poll `GET /v1/projects/{project_id}/documents` to track
/// indexing progress.
#[utoipa::path(
    post,
    path = "/v1/projects/{project_id}/documents/upload",
    tag = "documents",
    params(("project_id" = String, Path, description = "Project ID")),
    request_body(content = inline(crate::api::openapi::DocumentUpload), content_type = "multipart/form-data"),
    summary = "Upload a document",
    description = "",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn upload_document_file(
    Path(project_id): Path<String>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<Value>, HTTPException> {
    let form = read_form(multipart).await?;
    form.require(&["file", "cwd"])?;
    let file = form.get("file").cloned().unwrap_or_default();
    let cwd = form.get("cwd").map(|f| f.text()).unwrap_or_default();
    let model_service = get_model_service();

    let filename = match file.filename.as_deref() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => "upload".to_string(),
    };
    let result = run_on(&EXECUTOR, move || {
        model_service.upload_document_to_inbox(&project_id, &filename, &file.data, Some(&cwd))
    })
    .await?;
    match result {
        Ok(value) => Ok(Json(value)),
        Err(e) => match py_error_kind(&e) {
            PyErrorKind::Value => Err(HTTPException::new(404, e.to_string())),
            PyErrorKind::Runtime => Err(HTTPException::new(503, e.to_string())),
            _ => {
                tracing::error!(target: LOGGER, "Error uploading document file: {:?}", e);
                Err(HTTPException::new(500, e.to_string()))
            }
        },
    }
}

/// `DELETE /v1/projects/{project_id}/documents/{document_id}` — delete a document.
#[utoipa::path(
    delete,
    path = "/v1/projects/{project_id}/documents/{document_id}",
    tag = "documents",
    params(("project_id" = String, Path, description = "Project ID"), ("document_id" = String, Path, description = "Document ID")),
    summary = "Delete a document",
    description = "",
    responses((status = 200, description = "OK", body = DeleteDocumentResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn delete_document_from_project(
    Path((project_id, document_id)): Path<(String, String)>,
) -> Result<Json<DeleteDocumentResponse>, HTTPException> {
    let model_service = get_model_service();
    let result =
        run_on(&EXECUTOR, move || model_service.delete_document(&document_id, Some(&project_id))).await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => match py_error_kind(&e) {
            PyErrorKind::Value => Err(HTTPException::new(404, e.to_string())),
            PyErrorKind::Runtime => Err(HTTPException::new(503, e.to_string())),
            _ => {
                tracing::error!(target: LOGGER, "Error deleting document from project: {:?}", e);
                Err(HTTPException::new(500, e.to_string()))
            }
        },
    }
}
