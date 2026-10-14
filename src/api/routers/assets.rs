//! Port of `smartloop/api/routers/assets.py`.
//!
//! Asset management router — upload files and reference them in chat messages.

use axum::extract::multipart::MultipartRejection;
use axum::extract::{DefaultBodyLimit, Multipart, Path};
use axum::routing::{get, post};
use axum::{Json, Router};
use once_cell::sync::Lazy;
use tokio::sync::Semaphore;

use crate::api::model::{AssetResponse, AssetUploadResponse, DeleteAssetResponse};
use crate::api::services::get_model_service;

use super::_errors::{parse_model, py_error_kind, read_form, run_on, HTTPException, PyErrorKind};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.assets";

/// The router's `ThreadPoolExecutor(max_workers=2)`.
static EXECUTOR: Lazy<Semaphore> = Lazy::new(|| Semaphore::new(2));

/// Routes under `/v1` (tag `assets`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/assets", post(upload_asset))
        .route("/v1/assets/{asset_id}", get(get_asset).delete(delete_asset))
        // Starlette reads uploads of any size; the service decides what is too large (413).
        .layer(DefaultBodyLimit::disable())
}

/// `ValueError` → *value_status*, anything else → 500 (logged as *what*).
fn asset_error(err: anyhow::Error, value_status: u16, what: &str) -> HTTPException {
    if py_error_kind(&err) == PyErrorKind::Value {
        return HTTPException::new(value_status, err.to_string());
    }
    tracing::error!(target: LOGGER, "{}: {:?}", what, err);
    HTTPException::new(500, err.to_string())
}

/// `POST /v1/assets` — upload a file as a reusable asset.
///
/// Supported document types (pdf, csv, docx, pptx, xlsx, txt) are automatically
/// converted to markdown and stored for efficient injection into chat context.
/// Other file types are stored as-is.  Returns an `asset_id` that can be
/// referenced in `message.attachments` of a chat completion request.
/// A `ValueError` (too large) answers 413.
#[utoipa::path(
    post,
    path = "/v1/assets",
    tag = "assets",
    request_body(content = inline(crate::api::openapi::FileUpload), content_type = "multipart/form-data"),
    summary = "Upload an asset",
    description = "Uploads a chat attachment and converts it to Markdown where possible.",
    responses((status = 200, description = "OK", body = AssetUploadResponse), (status = 413, description = "File too large"), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn upload_asset(
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<AssetUploadResponse>, HTTPException> {
    let form = read_form(multipart).await?;
    form.require(&["file"])?;
    let file = form.get("file").cloned().unwrap_or_default();
    let model_service = get_model_service();

    let filename = match file.filename.as_deref() {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => "upload".to_string(),
    };
    let content_type = file.content_type.clone();
    let result = run_on(&EXECUTOR, move || {
        model_service.upload_asset(&file.data, &filename, content_type.as_deref())
    })
    .await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(asset_error(e, 413, "Error uploading asset")),
    }
}

/// `GET /v1/assets/{asset_id}` — retrieve asset metadata and content (markdown) by ID.
#[utoipa::path(
    get,
    path = "/v1/assets/{asset_id}",
    tag = "assets",
    params(("asset_id" = String, Path, description = "Asset ID")),
    summary = "Get an asset",
    description = "",
    responses((status = 200, description = "OK", body = AssetResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn get_asset(Path(asset_id): Path<String>) -> Result<Json<AssetResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_on(&EXECUTOR, move || model_service.get_asset(&asset_id)).await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(asset_error(e, 404, "Error retrieving asset")),
    }
}

/// `DELETE /v1/assets/{asset_id}` — delete an asset by ID.
#[utoipa::path(
    delete,
    path = "/v1/assets/{asset_id}",
    tag = "assets",
    params(("asset_id" = String, Path, description = "Asset ID")),
    summary = "Delete an asset",
    description = "",
    responses((status = 200, description = "OK", body = DeleteAssetResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn delete_asset(
    Path(asset_id): Path<String>,
) -> Result<Json<DeleteAssetResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_on(&EXECUTOR, move || model_service.delete_asset(&asset_id)).await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) => Err(asset_error(e, 404, "Error deleting asset")),
    }
}
