//! Port of `smartloop/api/routers/mcp_servers.py`.
//!
//! MCP server management router: list/inspect/refresh/delete project MCP
//! servers, unified registration (local stdio, Smartloop-hosted, remote with
//! token or OAuth), TOML drop-in definitions, and the OAuth redirect callback.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use axum::extract::{Path as AxumPath, Query};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use indexmap::IndexMap;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::api::model::mcp::{
    DeleteMCPServerResponse, MCPDefinitionRequest, MCPOAuthCallbackRequest, MCPRegisterRequest,
    MCPRegisterResponse, MCPServerResponse, MCPToolResponse,
};
pub use crate::api::routers::mcp_catalog::HTTPException;
#[allow(unused_imports)]
use crate::api::routers::mcp_catalog::{resolve_catalog_entry, MCPCatalogEntry};
#[allow(unused_imports)]
use crate::api::services::{get_model_service, model_service::ModelService};
#[allow(unused_imports)]
use crate::config::app_settings::AppSettings;
#[allow(unused_imports)]
use crate::mcp::client::{MCPClient, MCPConnectResult, MCPDiscoveryError, MCPServerInfo};
#[allow(unused_imports)]
use crate::mcp::oauth::{MCPOAuthError, MCPOAuthService, TokenResponse};

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOGGER: &str = "smartloop.api.routers.mcp_servers";

/// How long an un-redeemed OAuth session lives.  It holds a client_secret and
/// code_verifier, so an abandoned flow must not live for the process lifetime.
pub const OAUTH_SESSION_TTL_S: f64 = 600.0;
/// Completed callback responses are cached only to absorb the browser's
/// duplicate GET of the redirect, which arrives within seconds.
pub const OAUTH_RESPONSE_TTL_S: f64 = 120.0;
/// Cap on cached completed callback responses.
pub const OAUTH_RESPONSE_MAX: usize = 32;

/// A pending OAuth session (the Python free-form dict, keyed by field name).
pub type OAuthSession = Map<String, Value>;

/// In-memory store for pending MCP OAuth sessions, keyed by `state`.
pub static PENDING_OAUTH_SESSIONS: Lazy<Mutex<HashMap<String, OAuthSession>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Completed callback responses keyed by `state` (insertion-ordered, oldest
/// first): `(expires_at, response)`.
pub static COMPLETED_OAUTH_RESPONSES: Lazy<Mutex<IndexMap<String, (f64, HtmlPage)>>> =
    Lazy::new(|| Mutex::new(IndexMap::new()));

/// Single-worker gate standing in for the `ThreadPoolExecutor(max_workers=1)`:
/// registration work is serialized so two registrations can't race the
/// workspace record or the lazily-loaded model caches.
pub static EXECUTOR: Lazy<tokio::sync::Semaphore> = Lazy::new(|| tokio::sync::Semaphore::new(1));

/// An HTML response with a status code (FastAPI `HTMLResponse`).
#[derive(Debug, Clone, PartialEq)]
pub struct HtmlPage {
    pub status_code: u16,
    pub body: String,
}

impl HtmlPage {
    /// A page with *status_code*.
    pub fn new(body: impl Into<String>, status_code: u16) -> Self {
        Self { status_code, body: body.into() }
    }

    /// A `200 OK` page.
    pub fn ok(body: impl Into<String>) -> Self {
        Self::new(body, 200)
    }
}

impl IntoResponse for HtmlPage {
    fn into_response(self) -> Response {
        let status = StatusCode::from_u16(self.status_code).unwrap_or(StatusCode::OK);
        (status, Html(self.body)).into_response()
    }
}

/// What the OAuth redirect callback answers with: an HTML page, or a `302`
/// redirect (the invalid_scope retry).
#[derive(Debug, Clone, PartialEq)]
pub enum OAuthCallbackResponse {
    Html(HtmlPage),
    /// `RedirectResponse(url, status_code=302)`.
    Redirect(String),
}

impl OAuthCallbackResponse {
    /// HTTP status of the response.
    pub fn status_code(&self) -> u16 {
        todo!()
    }

    /// The HTML body (empty for a redirect).
    pub fn body(&self) -> &str {
        todo!()
    }

    /// The `Location` of a redirect.
    pub fn location(&self) -> Option<&str> {
        todo!()
    }
}

impl IntoResponse for OAuthCallbackResponse {
    fn into_response(self) -> Response {
        match self {
            OAuthCallbackResponse::Html(page) => page.into_response(),
            // FastAPI's RedirectResponse answers 302 Found.
            OAuthCallbackResponse::Redirect(url) => {
                (StatusCode::FOUND, [(axum::http::header::LOCATION, url)]).into_response()
            }
        }
    }
}

/// Keyword arguments to `MCPOAuthService.exchange_code_for_tokens`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenExchangeParams {
    pub token_endpoint: String,
    pub code: String,
    pub redirect_uri: String,
    pub client_id: String,
    pub code_verifier: String,
    pub client_secret: Option<String>,
    pub resource: Option<String>,
}

/// Test seam for the blocking OAuth token exchange (Python tests patch
/// `MCPOAuthService`).
pub type ExchangeOAuthCodeFn =
    Arc<dyn Fn(TokenExchangeParams) -> Result<TokenResponse, MCPOAuthError> + Send + Sync>;
/// Test seam for remote discovery: `(server_url, access_token, token_type)`.
pub type DiscoverRemoteToolsFn = Arc<
    dyn Fn(&str, Option<&str>, &str) -> anyhow::Result<MCPConnectResult> + Send + Sync,
>;
/// Test seam for the Smartloop `/info` probe: returns the parsed JSON.
pub type ProbeInfoFn = Arc<dyn Fn(&str) -> anyhow::Result<Value> + Send + Sync>;

/// Overridable hooks replacing the network-touching collaborators.
#[derive(Clone, Default)]
pub struct McpRouteHooks {
    pub exchange_oauth_code: Option<ExchangeOAuthCodeFn>,
    pub discover_remote_tools: Option<DiscoverRemoteToolsFn>,
    pub probe_info: Option<ProbeInfoFn>,
}

static HOOKS: Lazy<Mutex<McpRouteHooks>> = Lazy::new(|| Mutex::new(McpRouteHooks::default()));

/// Install (or with `McpRouteHooks::default()`, clear) the test hooks.
pub fn set_hooks(hooks: McpRouteHooks) {
    *HOOKS.lock() = hooks;
}

/// Query string of the GET OAuth callbacks.
#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct OAuthCallbackQuery {
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub iss: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub error_description: Option<String>,
    #[serde(default)]
    pub error_uri: Option<String>,
}

/// Drop expired pending OAuth sessions and cached callback responses.
///
/// Piggybacked on request handling rather than run as a background task: the
/// maps only grow when a registration happens, so sweeping on the same events
/// keeps them bounded.  A session without `created_at` is stamped with *now*
/// so it expires one TTL later.
pub fn purge_oauth_state(now: Option<f64>) {
    let _ = now;
    todo!()
}

/// Run a blocking call on the single registration worker and await it.
///
/// Every MCP service/network call is synchronous; left on the runtime they
/// stall every other request and open SSE stream.  Panics are resumed.
pub async fn offload<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let _ = f;
    todo!()
}

/// Blocking OAuth token exchange, for use via [`offload`].
pub fn exchange_oauth_code(params: TokenExchangeParams) -> Result<TokenResponse, MCPOAuthError> {
    let _ = params;
    todo!()
}

/// Blocking remote MCP handshake + tool discovery, for use via [`offload`].
pub fn discover_remote_tools(
    server_url: &str,
    access_token: Option<&str>,
    token_type: &str,
) -> anyhow::Result<MCPConnectResult> {
    let _ = (server_url, access_token, token_type);
    todo!()
}

/// Build a `server_info` dict from an `MCPServerInfo`, including description
/// and icons.
pub fn build_server_info_dict(si: Option<&MCPServerInfo>) -> Option<Map<String, Value>> {
    let _ = si;
    todo!()
}

/// Build an `MCPServerResponse` from a `_mcp_server_to_dict` result.
pub fn build_server_response(result: &Map<String, Value>) -> MCPServerResponse {
    let _ = result;
    todo!()
}

/// Render what the authorization server said it objected to (escaped HTML).
///
/// `error_uri` is only rendered as a link when it is an http(s) URL; anything
/// else (`javascript:` above all) is shown as inert text.
pub fn authorization_error_detail(
    error: &str,
    error_description: Option<&str>,
    error_uri: Option<&str>,
) -> String {
    let _ = (error, error_description, error_uri);
    todo!()
}

/// Refuse a redirect that did not come from the server we sent the user to
/// (RFC 9207 mix-up defence).  Returns the 400 page to answer with, or `None`
/// to proceed.
pub fn reject_unexpected_issuer(state: Option<&str>, iss: Option<&str>) -> Option<HtmlPage> {
    let _ = (state, iss);
    todo!()
}

/// Reissue the authorization once, asking for no scopes at all (after an
/// `invalid_scope` refusal).  Returns the `302` redirect URL, or `None`.
pub fn retry_authorization_without_scope(state: Option<&str>, error: &str) -> Option<String> {
    let _ = (state, error);
    todo!()
}

/// Render the "connected" page shown in the browser after OAuth, with an
/// optional "Return to Smartloop" link carrying `server_id`.
pub fn oauth_success_page(
    server_name: &str,
    tool_count: usize,
    server_id: &str,
    return_url: Option<&str>,
) -> HtmlPage {
    let _ = (server_name, tool_count, server_id, return_url);
    todo!()
}

/// Register a local (stdio) MCP server from a client-supplied command.
///
/// When *curated* is a resolved catalog entry, its config and metadata win
/// over the request.  Returns as soon as the connection is persisted, with
/// `status="pending"`.
pub async fn register_local_mcp_server(
    request: &MCPRegisterRequest,
    model_service: Arc<ModelService>,
    project_id: &str,
    curated: Option<&MCPCatalogEntry>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let _ = (request, model_service, project_id, curated);
    todo!()
}

/// Try to register a Smartloop-hosted MCP server by probing its `/info`
/// endpoint.  Returns `Ok(None)` if the URL doesn't expose one, allowing the
/// caller to fall through to the standard MCP handshake.
pub async fn try_register_smartloop_mcp_server(
    server_url: &str,
    model_service: Arc<ModelService>,
    project_id: &str,
) -> Result<Option<MCPRegisterResponse>, HTTPException> {
    let _ = (server_url, model_service, project_id);
    todo!()
}

/// Register a remote MCP server.  Tries direct connection first, falls back
/// to OAuth (stashing a pending session and returning `auth_type="oauth"`).
pub async fn register_remote_mcp_server(
    request: &MCPRegisterRequest,
    model_service: Arc<ModelService>,
    project_id: &str,
    curated: Option<&MCPCatalogEntry>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let _ = (request, model_service, project_id, curated);
    todo!()
}

/// `GET /v1/projects/{project_id}/mcp` — `{"servers": [...]}` with each
/// server's `indexing` block.
pub async fn list_mcp_servers_for_project(
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<Value>, HTTPException> {
    let _ = project_id;
    todo!()
}

/// `GET /v1/projects/{project_id}/mcp/tools/count` — diagnostic tool count.
pub async fn count_mcp_tools_for_project(
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<Value>, HTTPException> {
    let _ = project_id;
    todo!()
}

/// `GET /v1/projects/{project_id}/mcp/{server_id}` — one server with its
/// provisioning status.
pub async fn get_mcp_server(
    AxumPath((project_id, server_id)): AxumPath<(String, String)>,
) -> Result<Json<Value>, HTTPException> {
    let _ = (project_id, server_id);
    todo!()
}

/// `POST /v1/projects/{project_id}/mcp/{server_id}/refresh` — re-run
/// background provisioning for one server.
pub async fn refresh_mcp_server(
    AxumPath((project_id, server_id)): AxumPath<(String, String)>,
) -> Result<Json<MCPServerResponse>, HTTPException> {
    let _ = (project_id, server_id);
    todo!()
}

/// `DELETE /v1/projects/{project_id}/mcp/{server_id}`.
pub async fn delete_mcp_server_from_project(
    AxumPath((project_id, server_id)): AxumPath<(String, String)>,
) -> Result<Json<DeleteMCPServerResponse>, HTTPException> {
    let _ = (project_id, server_id);
    todo!()
}

/// Unified MCP server registration (project-scoped).
///
/// Resolves curated metadata server-side and dispatches on the resolved
/// entry's `server_type` (else the request's): `"local"` → stdio, anything
/// else → remote.
pub async fn register_mcp_server(
    project_id: &str,
    request: MCPRegisterRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let _ = (project_id, request, model_service);
    todo!()
}

/// Axum handler for `POST /v1/projects/{project_id}/mcp/register`.
pub async fn register_mcp_server_handler(
    AxumPath(project_id): AxumPath<String>,
    Json(request): Json<MCPRegisterRequest>,
) -> Result<Json<MCPRegisterResponse>, HTTPException> {
    let _ = (project_id, request);
    todo!()
}

/// Install a connection from a dropped-in TOML file.
///
/// The file is saved under `~/.smartloop/mcp.d/<key>.toml` (before
/// registering; a failure to save is logged and ignored) and the connection
/// goes through [`register_mcp_server`].
pub async fn register_mcp_from_definition(
    project_id: &str,
    request: MCPDefinitionRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let _ = (project_id, request, model_service);
    todo!()
}

/// Core of [`register_mcp_from_definition`] with the home dir and the
/// registration step injected (the Python tests patch `AppSettings` and
/// `register_mcp_server`).  *register* receives `(project_id, request)`.
pub async fn register_mcp_from_definition_with<F, Fut>(
    project_id: &str,
    request: MCPDefinitionRequest,
    home_dir: &Path,
    register: F,
) -> Result<MCPRegisterResponse, HTTPException>
where
    F: FnOnce(String, MCPRegisterRequest) -> Fut,
    Fut: Future<Output = Result<MCPRegisterResponse, HTTPException>>,
{
    let _ = (project_id, request, home_dir, register);
    todo!()
}

/// Axum handler for `POST /v1/projects/{project_id}/mcp/definition`.
pub async fn register_mcp_from_definition_handler(
    AxumPath(project_id): AxumPath<String>,
    Json(request): Json<MCPDefinitionRequest>,
) -> Result<Json<MCPRegisterResponse>, HTTPException> {
    let _ = (project_id, request);
    todo!()
}

/// `GET /v1/projects/{project_id}/mcp/oauth/callback` — where a provider's
/// redirect lands, scoped to the project it belongs to.
pub async fn mcp_oauth_callback_for_project(
    project_id: &str,
    query: OAuthCallbackQuery,
    model_service: Arc<ModelService>,
) -> OAuthCallbackResponse {
    let _ = (project_id, query, model_service);
    todo!()
}

/// `GET /v1/mcp/oauth/callback` — the unscoped redirect target.
pub async fn mcp_oauth_callback(
    query: OAuthCallbackQuery,
    model_service: Arc<ModelService>,
) -> OAuthCallbackResponse {
    let _ = (query, model_service);
    todo!()
}

async fn mcp_oauth_callback_for_project_handler(
    AxumPath(project_id): AxumPath<String>,
    Query(query): Query<OAuthCallbackQuery>,
) -> OAuthCallbackResponse {
    let _ = (project_id, query);
    todo!()
}

async fn mcp_oauth_callback_handler(Query(query): Query<OAuthCallbackQuery>) -> OAuthCallbackResponse {
    let _ = query;
    todo!()
}

/// Exchange the code for tokens, discover tools, and register the server.
pub async fn complete_oauth_redirect(
    query: OAuthCallbackQuery,
    model_service: Arc<ModelService>,
    routed_project_id: Option<&str>,
) -> OAuthCallbackResponse {
    let _ = (query, model_service, routed_project_id);
    todo!()
}

/// Where the provider sends the browser back when the client names nowhere:
/// the supervisor's public port, scoped to the project.
pub fn default_callback_url(project_id: &str) -> String {
    let _ = project_id;
    todo!()
}

/// Whether a server saved here would reach *project_id*'s own file.
pub fn can_persist_project(project_id: Option<&str>) -> bool {
    let _ = project_id;
    todo!()
}

/// The 500 page for a callback answered by a process that cannot save the
/// project.
pub fn wrong_process_page(project_id: Option<&str>) -> HtmlPage {
    let _ = project_id;
    todo!()
}

/// `POST /v1/projects/{project_id}/mcp/oauth/callback` — exchange an OAuth
/// authorization code for tokens and register the remote MCP server.
pub async fn complete_mcp_oauth(
    project_id: &str,
    request: MCPOAuthCallbackRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPServerResponse, HTTPException> {
    let _ = (project_id, request, model_service);
    todo!()
}

async fn complete_mcp_oauth_handler(
    AxumPath(project_id): AxumPath<String>,
    Json(request): Json<MCPOAuthCallbackRequest>,
) -> Result<Json<MCPServerResponse>, HTTPException> {
    let _ = (project_id, request);
    todo!()
}

/// The router (`APIRouter(prefix="/v1", tags=["mcp"])`).  Literal routes are
/// registered alongside the `{server_id}` ones; axum prefers static segments.
pub fn router() -> Router {
    Router::new()
        .route("/v1/projects/{project_id}/mcp", get(list_mcp_servers_for_project))
        .route("/v1/projects/{project_id}/mcp/tools/count", get(count_mcp_tools_for_project))
        .route(
            "/v1/projects/{project_id}/mcp/{server_id}",
            get(get_mcp_server).delete(delete_mcp_server_from_project),
        )
        .route("/v1/projects/{project_id}/mcp/{server_id}/refresh", post(refresh_mcp_server))
        .route("/v1/projects/{project_id}/mcp/register", post(register_mcp_server_handler))
        .route(
            "/v1/projects/{project_id}/mcp/definition",
            post(register_mcp_from_definition_handler),
        )
        .route(
            "/v1/projects/{project_id}/mcp/oauth/callback",
            get(mcp_oauth_callback_for_project_handler).post(complete_mcp_oauth_handler),
        )
        .route("/v1/mcp/oauth/callback", get(mcp_oauth_callback_handler))
}
