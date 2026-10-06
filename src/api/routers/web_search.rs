//! Port of `smartloop/api/routers/web_search.py`.
//!
//! Web search enablement (project-level).
//!
//! `PATCH /v1/projects/{id}/websearch` is the user's own switch, toggled by the
//! Cmd/Ctrl+Alt+S shortcut. It says nothing about whether a search can actually
//! run: that depends on the account being signed in with a provider registered,
//! which the hosted `GET /v1/websearch/status` answers and the orchestrator
//! checks before searching. There is deliberately no second local switch — it
//! would only duplicate what the endpoint already knows.

use axum::extract::Path;
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::api::services::get_model_service;

use super::_errors::{py_error_kind, run_blocking, unhandled_error, HTTPException, PyErrorKind, ValidatedJson};

/// Logger name used by this router.
pub const LOGGER: &str = "smartloop.api.routers.web_search";

/// Body of `PATCH /v1/projects/{id}/websearch`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WebSearchEnabledRequest {
    pub enabled: bool,
}

/// Web search state of a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct WebSearchConfigResponse {
    pub project_id: String,
    pub enabled: bool,
}

/// Routes under `/v1` (tag `web-search`).
pub fn router() -> Router {
    Router::new().route(
        "/v1/projects/{project_id}/websearch",
        get(get_web_search_config).patch(set_web_search_enabled),
    )
}

/// Turn the service's answer into the response model, or its failure into
/// the router's status: `ValueError` → 404, anything else → 500 (logged).
///
/// The response model is validated after the handler returned, outside its
/// `try`, so a malformed answer is FastAPI's bare 500.
fn respond(result: anyhow::Result<Value>, what: &str) -> Result<Json<WebSearchConfigResponse>, HTTPException> {
    match result {
        Ok(value) => serde_json::from_value(value).map(Json).map_err(|e| {
            tracing::error!(target: LOGGER, "Invalid web search config response: {}", e);
            unhandled_error()
        }),
        Err(e) if py_error_kind(&e) == PyErrorKind::Value => Err(HTTPException::new(404, e.to_string())),
        Err(e) => {
            tracing::error!(target: LOGGER, "{}: {:?}", what, e);
            Err(HTTPException::new(500, e.to_string()))
        }
    }
}

/// `GET /v1/projects/{project_id}/websearch` — whether web search is enabled.
pub async fn get_web_search_config(
    Path(project_id): Path<String>,
) -> Result<Json<WebSearchConfigResponse>, HTTPException> {
    let model_service = get_model_service();
    let result = run_blocking(move || model_service.get_web_search_config(&project_id)).await?;
    respond(result, "Error reading web search config")
}

/// `PATCH /v1/projects/{project_id}/websearch` — enable/disable web search (Pro feature).
pub async fn set_web_search_enabled(
    Path(project_id): Path<String>,
    ValidatedJson(request): ValidatedJson<WebSearchEnabledRequest>,
) -> Result<Json<WebSearchConfigResponse>, HTTPException> {
    let model_service = get_model_service();
    let result =
        run_blocking(move || model_service.set_web_search_enabled(&project_id, request.enabled)).await?;
    respond(result, "Error updating web search config")
}
