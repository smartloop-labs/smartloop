//! Port of `smartloop/api/routers/feedback.py`.
//!
//! Feedback router for thumbs up/down on conversation responses.

use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::api::model::{FeedbackRequest, FeedbackResponse};
use crate::api::services::get_model_service;

use super::_errors::{
    parse_model, py_error_kind, run_blocking, HTTPException, PyErrorKind, ValidatedJson, ValidatedQuery,
};

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

/// The project id as the service's `str` parameter receives it.  The Python
/// router passed a missing id through as `None`, which the service then
/// formats into its "Project not found: None" error.
fn project_arg(project_id: Option<&str>) -> String {
    project_id.unwrap_or("None").to_string()
}

/// `GET /v1/feedback` — return a map of message_id -> rating for a session.
///
/// A `ValueError` (model or project not loaded yet) returns `{}`.
pub async fn get_feedback(
    ValidatedQuery(query): ValidatedQuery<GetFeedbackQuery>,
) -> Result<Json<Map<String, Value>>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        let project_id = project_arg(query.project_id.as_deref());
        model_service.get_feedback_ratings(&query.session_id, &project_id)
    })
    .await?;
    match result {
        Ok(ratings) => Ok(Json(
            ratings.into_iter().map(|(message_id, rating)| (message_id, Value::String(rating))).collect(),
        )),
        // Model or project not loaded yet — return empty feedback
        Err(e) if py_error_kind(&e) == PyErrorKind::Value => Ok(Json(Map::new())),
        Err(e) => {
            tracing::error!(target: LOGGER, "Error loading feedback: {:?}", e);
            Err(HTTPException::new(500, e.to_string()))
        }
    }
}

/// `POST /v1/feedback` — record a rating; `ValueError` answers 400.
pub async fn submit_feedback(
    ValidatedJson(request): ValidatedJson<FeedbackRequest>,
) -> Result<Json<FeedbackResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        let project_id = project_arg(request.project_id.as_deref());
        model_service.submit_feedback(
            &request.session_id,
            &request.message_id,
            &project_id,
            request.rating.as_str(),
        )
    })
    .await?;
    match result {
        Ok(value) => parse_model(value).map(Json),
        Err(e) if py_error_kind(&e) == PyErrorKind::Value => {
            tracing::warn!(target: LOGGER, "[feedback] Bad request: {}", e);
            Err(HTTPException::new(400, e.to_string()))
        }
        Err(e) => {
            tracing::error!(target: LOGGER, "Error submitting feedback: {:?}", e);
            Err(HTTPException::new(500, e.to_string()))
        }
    }
}

/// `DELETE /v1/feedback` — delete feedback ratings for a message ID.
pub async fn delete_feedback(
    ValidatedQuery(query): ValidatedQuery<DeleteFeedbackQuery>,
) -> Result<Json<Value>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || {
        model_service.delete_feedback_ratings(&query.session_id, &query.message_id, query.project_id.as_deref())
    })
    .await?;
    match result {
        Ok(()) => Ok(Json(json!({"detail": "Feedback deleted successfully"}))),
        Err(e) => {
            tracing::error!(target: LOGGER, "Error deleting feedback: {:?}", e);
            Err(HTTPException::new(500, e.to_string()))
        }
    }
}
