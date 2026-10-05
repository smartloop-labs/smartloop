//! Port of `smartloop/api/routers/mcp_catalog.py`.
//!
//! MCP server catalog router.
//!
//! Fetches the curated MCP server catalog from the Smartloop platform API and
//! exposes it to local clients (TUI, Electron) for one-click install.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[allow(unused_imports)]
use crate::platform::{get_platform_service, PlatformError, PlatformService};

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOGGER: &str = "smartloop.api.routers.mcp_catalog";

/// FastAPI's `HTTPException`: a status code plus a `detail` that renders as
/// `{"detail": ...}`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{status_code}: {detail}")]
pub struct HTTPException {
    /// HTTP status code.
    pub status_code: u16,
    /// The `detail` field of the error body.
    pub detail: String,
}

impl HTTPException {
    /// Build an exception with *status_code* and *detail*.
    pub fn new(status_code: u16, detail: impl Into<String>) -> Self {
        Self { status_code, detail: detail.into() }
    }
}

impl IntoResponse for HTTPException {
    fn into_response(self) -> Response {
        let status =
            StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        (status, Json(serde_json::json!({ "detail": self.detail }))).into_response()
    }
}

/// Connection settings for a remotely hosted catalog server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct RemoteConfig {
    pub server_url: String,
    #[serde(default)]
    pub auth_type: Option<String>,
    #[serde(default)]
    pub oauth_metadata_url: Option<String>,
    #[serde(default)]
    pub authorization_endpoint: Option<String>,
    #[serde(default)]
    pub token_endpoint: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
}

/// Launch settings for a catalog server run locally over stdio.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct LocalConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

/// One curated MCP catalog entry.
///
/// The platform catalog does not store `server_type`; [`MCPCatalogEntry::from_value`]
/// (the `model_validator`) derives it so clients still get a populated
/// `"local"`/`"remote"` value.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPCatalogEntry {
    pub slug: String,
    pub display_name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub server_type: Option<String>,
    /// `"external"`, `"internal"`, or `"local"`.
    pub hosted_by: String,
    #[serde(default)]
    pub remote_config: Option<RemoteConfig>,
    #[serde(default)]
    pub local_config: Option<LocalConfig>,
    #[serde(default)]
    pub icon_url: Option<String>,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub sort_order: i64,
    #[serde(default = "default_catalog_version")]
    pub catalog_version: i64,
}

fn default_catalog_version() -> i64 {
    1
}

impl MCPCatalogEntry {
    /// Validate a raw platform dict into an entry, applying the
    /// `server_type` default (`_default_server_type`).
    pub fn from_value(value: Value) -> Result<Self, serde_json::Error> {
        let _ = value;
        todo!()
    }

    /// The `model_validator(mode="after")`: fill `server_type` with `"local"`
    /// when a local config is present or the entry is hosted locally, else
    /// `"remote"`.
    pub fn default_server_type(mut self) -> Self {
        let _ = &mut self;
        todo!()
    }
}

/// Validate raw catalog dicts into entries (`_to_entries`).
pub fn to_entries(raw: Vec<serde_json::Map<String, Value>>) -> Vec<MCPCatalogEntry> {
    let _ = raw;
    todo!()
}

/// `GET /v1/mcp-catalog` — list curated MCP server catalog entries.
///
/// Clients (TUI, Electron) use this to populate the one-click install catalog.
/// Served through the cache-through platform service, so the list survives a
/// platform outage or an expired token instead of leaving the panel empty.
/// Permission errors → 401 with the message; connection errors → 502
/// "Platform API unavailable".  The platform client is blocking, so this runs
/// on the blocking pool.
pub async fn list_mcp_catalog() -> Result<Json<Vec<MCPCatalogEntry>>, HTTPException> {
    todo!()
}

/// Core of [`list_mcp_catalog`] against an explicit service (the FastAPI
/// `Depends(get_platform_service)` seam).
pub fn list_mcp_catalog_with(
    svc: &Arc<PlatformService>,
) -> Result<Vec<MCPCatalogEntry>, HTTPException> {
    let _ = svc;
    todo!()
}

/// Look up one curated entry by slug, or `None` if it can't be resolved.
///
/// Never fails: the caller falls back to the client-supplied server_url and
/// lets enrichment run, so an install is never blocked by a catalog lookup.
/// The platform exposes a list endpoint only, so this is fetch-and-find over
/// the cached list.
pub fn resolve_catalog_entry(slug: &str) -> Option<MCPCatalogEntry> {
    let _ = slug;
    todo!()
}

/// The router (`APIRouter(prefix="/v1", tags=["mcp-catalog"])`).
pub fn router() -> Router {
    Router::new().route("/v1/mcp-catalog", get(list_mcp_catalog))
}
