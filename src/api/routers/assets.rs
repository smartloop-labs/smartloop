//! Port of `smartloop/api/routers/assets.py`.
//!
//! Asset management router — upload files and reference them in chat messages.

use axum::extract::{Multipart, Path};
use axum::routing::{get, post};
use axum::{Json, Router};

use crate::api::model::{AssetResponse, AssetUploadResponse, DeleteAssetResponse};
#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.assets";

/// Routes under `/v1` (tag `assets`).
pub fn router() -> Router {
    Router::new()
        .route("/v1/assets", post(upload_asset))
        .route("/v1/assets/{asset_id}", get(get_asset).delete(delete_asset))
}

/// `POST /v1/assets` — upload a file as a reusable asset.
///
/// Supported document types (pdf, csv, docx, pptx, xlsx, txt) are automatically
/// converted to markdown and stored for efficient injection into chat context.
/// Other file types are stored as-is.  Returns an `asset_id` that can be
/// referenced in `message.attachments` of a chat completion request.
/// A `ValueError` (too large) answers 413.
pub async fn upload_asset(multipart: Multipart) -> Result<Json<AssetUploadResponse>, HTTPException> {
    let _ = multipart;
    todo!()
}

/// `GET /v1/assets/{asset_id}` — retrieve asset metadata and content (markdown) by ID.
pub async fn get_asset(Path(asset_id): Path<String>) -> Result<Json<AssetResponse>, HTTPException> {
    let _ = asset_id;
    todo!()
}

/// `DELETE /v1/assets/{asset_id}` — delete an asset by ID.
pub async fn delete_asset(
    Path(asset_id): Path<String>,
) -> Result<Json<DeleteAssetResponse>, HTTPException> {
    let _ = asset_id;
    todo!()
}
