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

#[allow(unused_imports)]
use crate::api::services::{get_model_service, ModelService};

use super::_errors::HTTPException;

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

/// `GET /v1/projects/{project_id}/websearch` — whether web search is enabled.
pub async fn get_web_search_config(
    Path(project_id): Path<String>,
) -> Result<Json<WebSearchConfigResponse>, HTTPException> {
    let _ = project_id;
    todo!()
}

/// `PATCH /v1/projects/{project_id}/websearch` — enable/disable web search (Pro feature).
pub async fn set_web_search_enabled(
    Path(project_id): Path<String>,
    Json(request): Json<WebSearchEnabledRequest>,
) -> Result<Json<WebSearchConfigResponse>, HTTPException> {
    let _ = (project_id, request);
    todo!()
}
