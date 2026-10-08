//! Port of `smartloop/api/routers/mcp_catalog.py`.
//!
//! MCP server catalog router.
//!
//! Fetches the curated MCP server catalog from the Smartloop platform API and
//! exposes it to local clients (TUI, Electron) for one-click install.

use std::sync::Arc;

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::platform::{get_platform_service, PlatformService};

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

/// One value the client collects before registering the server.
///
/// Used by entries that connect with a pasted credential (e.g. a GitHub PAT)
/// instead of OAuth.  The client renders a form from these fields.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, utoipa::ToSchema)]
pub struct CredentialField {
    /// The `MCPRegisterRequest` field to send the value as, e.g. "token".
    pub key: String,
    pub label: String,
    /// What to paste and, for a token, which scopes it needs.
    #[serde(default)]
    pub help: Option<String>,
    /// Where to create the credential, e.g. GitHub's token settings page.
    #[serde(default)]
    pub help_url: Option<String>,
    /// Mask on entry and never echo back.
    #[serde(default = "default_true")]
    pub secret: bool,
    #[serde(default = "default_true")]
    pub required: bool,
}

fn default_true() -> bool {
    true
}

/// Connection settings for a remotely hosted catalog server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct RemoteConfig {
    pub server_url: String,
    /// "oauth", "token" or "none".  A "token" entry is one the client must
    /// prompt for; it carries `credential_fields` describing what to ask.
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
    #[serde(default)]
    pub credential_fields: Vec<CredentialField>,
}

/// Launch settings for a catalog server run locally over stdio.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
pub struct LocalConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Insertion-ordered, like the Python dict it mirrors.
    #[serde(default)]
    pub env: IndexMap<String, String>,
}

/// One curated MCP catalog entry.
///
/// The platform catalog does not store `server_type`; [`MCPCatalogEntry::from_value`]
/// (the `model_validator`) derives it so clients still get a populated
/// `"local"`/`"remote"` value.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, utoipa::ToSchema)]
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
        let entry: MCPCatalogEntry = serde_json::from_value(value)?;
        Ok(entry.default_server_type())
    }

    /// The `model_validator(mode="after")`: fill `server_type` with `"local"`
    /// when a local config is present or the entry is hosted locally, else
    /// `"remote"`.
    pub fn default_server_type(mut self) -> Self {
        let missing = self.server_type.as_deref().map(str::is_empty).unwrap_or(true);
        if missing {
            let derived = if self.local_config.is_some() || self.hosted_by == "local" {
                "local"
            } else {
                "remote"
            };
            self.server_type = Some(derived.to_string());
        }
        self
    }
}

/// Validate raw catalog dicts into entries (`_to_entries`).
///
/// Fails, like the pydantic constructor, on the first entry that does not
/// validate.
pub fn to_entries(
    raw: Vec<serde_json::Map<String, Value>>,
) -> Result<Vec<MCPCatalogEntry>, serde_json::Error> {
    raw.into_iter().map(|e| MCPCatalogEntry::from_value(Value::Object(e))).collect()
}

/// `GET /v1/mcp-catalog` — list curated MCP server catalog entries.
///
/// Clients (TUI, Electron) use this to populate the one-click install catalog.
/// Served through the cache-through platform service, so the list survives a
/// platform outage or an expired token instead of leaving the panel empty.
/// Permission errors → 401 with the message; connection errors → 502
/// "Platform API unavailable".
///
/// Blocking on purpose: the platform client is synchronous, and a 401 now
/// costs a refresh and a retry on top of the original call.  On the runtime
/// that would stall every in-flight chat stream, so it runs on the blocking
/// pool (FastAPI ran the sync handler in its threadpool).
#[utoipa::path(
    get,
    path = "/v1/mcp-catalog",
    tag = "mcp",
    summary = "List the curated MCP catalog",
    description = "",
    responses((status = 200, description = "OK", body = Vec<MCPCatalogEntry>), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_mcp_catalog() -> Result<Json<Vec<MCPCatalogEntry>>, HTTPException> {
    let joined = tokio::task::spawn_blocking(|| {
        let svc = get_platform_service();
        list_mcp_catalog_with(&svc)
    })
    .await;
    match joined {
        Ok(result) => result.map(Json),
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        Err(e) => Err(HTTPException::new(500, e.to_string())),
    }
}

/// Core of [`list_mcp_catalog`] against an explicit service (the FastAPI
/// `Depends(get_platform_service)` seam).
pub fn list_mcp_catalog_with(
    svc: &Arc<PlatformService>,
) -> Result<Vec<MCPCatalogEntry>, HTTPException> {
    let entries = match svc.list_mcp_catalog(false) {
        Ok(entries) => entries,
        Err(e) if e.is_permission() => return Err(HTTPException::new(401, e.message())),
        Err(e) if e.is_connection() => {
            return Err(HTTPException::new(502, "Platform API unavailable"));
        }
        Err(e) => {
            // Anything else escaped the Python handler: an unhandled 500.
            tracing::error!(target: LOGGER, "Error listing MCP catalog: {}", e);
            return Err(HTTPException::new(500, "Internal Server Error"));
        }
    };
    to_entries(entries).map_err(|e| {
        // A response-model validation failure is a server error.
        tracing::error!(target: LOGGER, "Invalid MCP catalog entry: {}", e);
        HTTPException::new(500, "Internal Server Error")
    })
}

/// Look up one curated entry by slug, or `None` if it can't be resolved.
///
/// Never fails: the caller falls back to the client-supplied server_url and
/// lets enrichment run, so an install is never blocked by a catalog lookup.
/// The platform exposes a list endpoint only, so this is fetch-and-find over
/// the cached list.
pub fn resolve_catalog_entry(slug: &str) -> Option<MCPCatalogEntry> {
    if slug.is_empty() {
        return None;
    }
    let entries = match get_platform_service().list_mcp_catalog(false) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(target: LOGGER, "Catalog lookup failed for slug '{}': {}", slug, e);
            return None;
        }
    };
    let entries = match to_entries(entries) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(target: LOGGER, "Catalog lookup failed for slug '{}': {}", slug, e);
            return None;
        }
    };
    if let Some(entry) = entries.into_iter().find(|entry| entry.slug == slug) {
        return Some(entry);
    }
    tracing::warn!(target: LOGGER, "No catalog entry for slug '{}'", slug);
    None
}

/// The router (`APIRouter(prefix="/v1", tags=["mcp-catalog"])`).
pub fn router() -> Router {
    Router::new().route("/v1/mcp-catalog", get(list_mcp_catalog))
}
