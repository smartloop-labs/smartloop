//! Port of `smartloop/api/routers/feedback.py`.
//!
//! Feedback router for thumbs up/down on conversation responses.

use axum::extract::Query;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::api::model::{FeedbackRequest, FeedbackResponse};
#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};

use super::_errors::HTTPException;

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.feedback";

/// Query for `GET /v1/feedback`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct GetFeedbackQuery {
    /// Session ID to load feedback for.
    pub session_id: String,
    /// Project ID.
    #[serde(default)]
    pub project_id: Option<String>,
}

/// Query for `DELETE /v1/feedback`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DeleteFeedbackQuery {
    /// Session ID to delete feedback for.
    pub session_id: String,
    /// Message ID to delete feedback for.
    pub message_id: String,
    /// Project ID.
    #[serde(default)]
    pub project_id: Option<String>,
}

/// Routes under `/v1` (tag `feedback`).
pub fn router() -> Router {
    Router::new().route(
        "/v1/feedback",
        get(get_feedback).post(submit_feedback).delete(delete_feedback),
    )
}

/// `GET /v1/feedback` — return a map of message_id -> rating for a session.
///
/// A `ValueError` (model or project not loaded yet) returns `{}`.
pub async fn get_feedback(
    Query(query): Query<GetFeedbackQuery>,
) -> Result<Json<Map<String, Value>>, HTTPException> {
    let _ = query;
    todo!()
}

/// `POST /v1/feedback` — record a rating; `ValueError` answers 400.
pub async fn submit_feedback(
    Json(request): Json<FeedbackRequest>,
) -> Result<Json<FeedbackResponse>, HTTPException> {
    let _ = request;
    todo!()
}

/// `DELETE /v1/feedback` — delete feedback ratings for a message ID.
pub async fn delete_feedback(
    Query(query): Query<DeleteFeedbackQuery>,
) -> Result<Json<Value>, HTTPException> {
    let _ = query;
    todo!()
}
