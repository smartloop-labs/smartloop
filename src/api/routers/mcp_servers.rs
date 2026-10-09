//! Port of `smartloop/api/routers/mcp_servers.py`.
//!
//! MCP server management router: list/inspect/refresh/delete project MCP
//! servers, unified registration (local stdio, Smartloop-hosted, remote with
//! token or OAuth), TOML drop-in definitions, and the OAuth redirect callback.

use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::{Path as AxumPath, Query};
use axum::body::Bytes;
use axum::http::header::CONTENT_TYPE;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post, MethodRouter};
use axum::{Json, Router};
use indexmap::IndexMap;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::api::model::mcp::{
    DeleteMCPServerResponse, MCPDefinitionRequest, MCPDevicePollRequest, MCPDevicePollResponse,
    MCPDeviceStartRequest, MCPDeviceStartResponse, MCPOAuthCallbackRequest, MCPRegisterRequest,
    MCPRegisterResponse, MCPServerResponse,
};
pub use crate::api::routers::mcp_catalog::HTTPException;
use crate::api::routers::mcp_catalog::{resolve_catalog_entry, MCPCatalogEntry};
use crate::api::services::model_service::{ModelService, ModelServiceError};
use crate::api::services::get_model_service;
use crate::config::app_settings::AppSettings;
use crate::mcp::client::{
    MCPAuthExpiredError, MCPClient, MCPClientError, MCPConnectResult, MCPDiscoveryError,
    MCPServerInfo,
};
use crate::mcp::definition_loader::{parse_definitions, reload_definitions, USER_DEFINITIONS_DIR};
use crate::mcp::definitions::MCPDefinitionError;
use crate::mcp::oauth::{
    ClientRegistrationResponse, DeviceAuthorization, DevicePoll, MCPOAuthError, MCPOAuthService,
    OAuthMetadata, ProtectedResourceMetadata, TokenResponse,
};
use crate::workspace::context::{owned_project_id, pinned_project_id, projects_delegated};

/// Python logger name (`logging.getLogger(__name__)`).
pub const LOGGER: &str = "smartloop.api.routers.mcp_servers";

// In-memory store for pending MCP OAuth sessions (keyed by state).  Entries are
// popped by the callback, but an abandoned flow (user closes the browser tab)
// never gets one — and the session holds a client_secret and code_verifier, so
// it must not live for the life of the process.  Hence the TTL sweep below.

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

/// A device sign-in waiting on the user to enter its code.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingDeviceSession {
    pub project_id: String,
    pub catalog_slug: String,
    /// The entry's URL at start, for registering if the catalog misses later.
    pub server_url: String,
    pub client_id: String,
    pub token_endpoint: String,
    pub device_code: String,
    /// Seconds between polls; grows on `slow_down`.
    pub interval: i64,
    /// Polls before this answer "pending" without asking the provider.
    pub next_poll_at: f64,
    pub expires_at: f64,
}

/// Pending device sign-ins, keyed by the session id handed to the client.
pub static PENDING_DEVICE_SESSIONS: Lazy<Mutex<HashMap<String, PendingDeviceSession>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Single-worker gate standing in for the `ThreadPoolExecutor(max_workers=1)`.
///
/// Registration spawns a subprocess, waits on the MCP handshake, may run local
/// inference and an embedding pass — all synchronous.  A single worker keeps
/// that off the runtime without letting two registrations race the workspace
/// record or the lazily-loaded model caches.
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
        match self {
            OAuthCallbackResponse::Html(page) => page.status_code,
            OAuthCallbackResponse::Redirect(_) => 302,
        }
    }

    /// The HTML body (empty for a redirect).
    pub fn body(&self) -> &str {
        match self {
            OAuthCallbackResponse::Html(page) => &page.body,
            OAuthCallbackResponse::Redirect(_) => "",
        }
    }

    /// The `Location` of a redirect.
    pub fn location(&self) -> Option<&str> {
        match self {
            OAuthCallbackResponse::Html(_) => None,
            OAuthCallbackResponse::Redirect(url) => Some(url),
        }
    }
}

impl From<HtmlPage> for OAuthCallbackResponse {
    fn from(page: HtmlPage) -> Self {
        OAuthCallbackResponse::Html(page)
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

/// Keyword arguments to `MCPOAuthService.build_authorization_url`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuthorizationUrlParams {
    pub authorization_endpoint: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub state: String,
    pub code_challenge: String,
    pub scope: Option<String>,
    pub resource: Option<String>,
}

/// The `MCPOAuthService` operations this router uses, so a test can stand in
/// for the whole service (the Python tests patch `smartloop.mcp.MCPOAuthService`).
///
/// The non-network helpers default to the real implementations.
pub trait OAuthFlow: Send {
    /// RFC 9728 protected-resource metadata discovery.
    fn discover_protected_resource_metadata(
        &mut self,
        server_url: &str,
        metadata_url: Option<&str>,
    ) -> Result<ProtectedResourceMetadata, MCPOAuthError>;

    /// RFC 8414 / OIDC authorization-server metadata discovery.
    fn discover_oauth_metadata(&mut self, server_url: &str) -> Result<OAuthMetadata, MCPOAuthError>;

    /// RFC 7591 dynamic client registration.
    fn register_client(
        &mut self,
        registration_endpoint: &str,
        redirect_uri: &str,
    ) -> Result<ClientRegistrationResponse, MCPOAuthError>;

    /// Redeem an authorization code at the token endpoint.
    fn exchange_code_for_tokens(
        &mut self,
        params: &TokenExchangeParams,
    ) -> Result<TokenResponse, MCPOAuthError>;

    /// RFC 8628 device authorization request.
    fn request_device_code(
        &mut self,
        device_authorization_endpoint: &str,
        client_id: &str,
        scope: Option<&str>,
    ) -> Result<DeviceAuthorization, MCPOAuthError>;

    /// One RFC 8628 device-token poll.
    fn poll_device_token(
        &mut self,
        token_endpoint: &str,
        client_id: &str,
        device_code: &str,
    ) -> Result<DevicePoll, MCPOAuthError>;

    /// Build the authorization redirect URL.
    fn build_authorization_url(&mut self, params: &AuthorizationUrlParams) -> String {
        MCPOAuthService::new(None).build_authorization_url(
            &params.authorization_endpoint,
            &params.client_id,
            &params.redirect_uri,
            &params.state,
            &params.code_challenge,
            params.scope.as_deref(),
            params.resource.as_deref(),
        )
    }

    /// RFC 8707 canonical resource indicator for *server_url*.
    fn canonical_resource(&self, server_url: &str) -> String {
        MCPOAuthService::canonical_resource(server_url)
    }

    /// `(code_verifier, code_challenge)` for PKCE S256.
    fn generate_pkce(&self) -> (String, String) {
        MCPOAuthService::generate_pkce()
    }

    /// A random `state` value.
    fn generate_state(&self) -> String {
        MCPOAuthService::generate_state()
    }
}

/// The real [`OAuthFlow`]: an `MCPOAuthService` used as a context manager.
///
/// The HTTP client is opened on the first network call (`__enter__`) and
/// closed on drop (`__exit__`), so building an authorization URL never opens
/// one — which also keeps a blocking client from being created or dropped on
/// the async runtime.
struct ServiceOAuthFlow {
    service: MCPOAuthService,
    entered: bool,
}

impl ServiceOAuthFlow {
    fn new() -> Self {
        Self { service: MCPOAuthService::new(None), entered: false }
    }

    fn entered(&mut self) -> &MCPOAuthService {
        if !self.entered {
            self.service.enter();
            self.entered = true;
        }
        &self.service
    }
}

impl Drop for ServiceOAuthFlow {
    fn drop(&mut self) {
        if self.entered {
            self.service.exit();
        }
    }
}

impl OAuthFlow for ServiceOAuthFlow {
    fn discover_protected_resource_metadata(
        &mut self,
        server_url: &str,
        metadata_url: Option<&str>,
    ) -> Result<ProtectedResourceMetadata, MCPOAuthError> {
        self.entered().discover_protected_resource_metadata(server_url, metadata_url)
    }

    fn discover_oauth_metadata(&mut self, server_url: &str) -> Result<OAuthMetadata, MCPOAuthError> {
        self.entered().discover_oauth_metadata(server_url)
    }

    fn register_client(
        &mut self,
        registration_endpoint: &str,
        redirect_uri: &str,
    ) -> Result<ClientRegistrationResponse, MCPOAuthError> {
        self.entered().register_client(registration_endpoint, redirect_uri, "Smartloop")
    }

    fn exchange_code_for_tokens(
        &mut self,
        params: &TokenExchangeParams,
    ) -> Result<TokenResponse, MCPOAuthError> {
        self.entered().exchange_code_for_tokens(
            &params.token_endpoint,
            &params.code,
            &params.redirect_uri,
            &params.client_id,
            &params.code_verifier,
            params.client_secret.as_deref(),
            params.resource.as_deref(),
        )
    }

    fn request_device_code(
        &mut self,
        device_authorization_endpoint: &str,
        client_id: &str,
        scope: Option<&str>,
    ) -> Result<DeviceAuthorization, MCPOAuthError> {
        self.entered().request_device_code(device_authorization_endpoint, client_id, scope)
    }

    fn poll_device_token(
        &mut self,
        token_endpoint: &str,
        client_id: &str,
        device_code: &str,
    ) -> Result<DevicePoll, MCPOAuthError> {
        self.entered().poll_device_token(token_endpoint, client_id, device_code)
    }

    fn build_authorization_url(&mut self, params: &AuthorizationUrlParams) -> String {
        self.service.build_authorization_url(
            &params.authorization_endpoint,
            &params.client_id,
            &params.redirect_uri,
            &params.state,
            &params.code_challenge,
            params.scope.as_deref(),
            params.resource.as_deref(),
        )
    }
}

/// Keyword arguments to `ModelService.add_local_mcp_server`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AddLocalMcpServer {
    pub command: String,
    pub args: Option<Vec<String>>,
    pub name: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub cwd: Option<String>,
    pub project_id: Option<String>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub catalog_slug: Option<String>,
}

/// Keyword arguments to `ModelService.add_remote_mcp_server`.
#[derive(Debug, Clone, PartialEq)]
pub struct AddRemoteMcpServer {
    pub name: String,
    pub server_url: String,
    pub auth_type: String,
    pub tools: Option<Vec<Value>>,
    pub token: Option<String>,
    pub token_endpoint: Option<String>,
    pub authorization_endpoint: Option<String>,
    pub resource: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub access_token: Option<String>,
    pub refresh_token: Option<String>,
    pub expires_at: Option<f64>,
    pub token_type: String,
    pub scope: Option<String>,
    pub project_id: Option<String>,
    pub server_info: Option<Map<String, Value>>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub catalog_slug: Option<String>,
    pub pkg: Option<String>,
}

impl Default for AddRemoteMcpServer {
    fn default() -> Self {
        Self {
            name: String::new(),
            server_url: String::new(),
            auth_type: "none".to_string(),
            tools: None,
            token: None,
            token_endpoint: None,
            authorization_endpoint: None,
            resource: None,
            client_id: None,
            client_secret: None,
            access_token: None,
            refresh_token: None,
            expires_at: None,
            token_type: "Bearer".to_string(),
            scope: None,
            project_id: None,
            server_info: None,
            description: None,
            tags: None,
            catalog_slug: None,
            pkg: None,
        }
    }
}

/// The `ModelService` MCP operations this router calls.  Implemented for
/// [`ModelService`]; a test installs its own through
/// [`McpRouteHooks::mcp_registry`] (the Python tests pass a `MagicMock`).
///
/// Every method blocks; the router calls them off the runtime.
pub trait McpServerRegistry: Send + Sync {
    fn add_local_mcp_server(&self, params: AddLocalMcpServer) -> anyhow::Result<Value>;
    fn add_remote_mcp_server(&self, params: AddRemoteMcpServer) -> anyhow::Result<Value>;
    fn list_mcp_servers_with_status(&self, project_id: &str) -> anyhow::Result<Vec<Value>>;
    fn count_mcp_tools(&self, project_id: &str) -> anyhow::Result<Value>;
    fn mcp_server_status(&self, server_id: &str, project_id: &str) -> anyhow::Result<Value>;
    fn retry_mcp_server(&self, server_id: &str, project_id: &str) -> anyhow::Result<Value>;
    fn delete_mcp_server(&self, server_id: &str, project_id: &str) -> anyhow::Result<Value>;
}

impl McpServerRegistry for ModelService {
    fn add_local_mcp_server(&self, p: AddLocalMcpServer) -> anyhow::Result<Value> {
        ModelService::add_local_mcp_server(
            self,
            &p.command,
            p.args,
            p.name.as_deref(),
            p.env,
            p.cwd.as_deref(),
            p.description.as_deref(),
            p.tags,
            p.catalog_slug.as_deref(),
            p.project_id.as_deref(),
        )
    }

    fn add_remote_mcp_server(&self, p: AddRemoteMcpServer) -> anyhow::Result<Value> {
        ModelService::add_remote_mcp_server(
            self,
            &p.name,
            &p.server_url,
            &p.auth_type,
            p.tools,
            p.token.as_deref(),
            p.token_endpoint.as_deref(),
            p.authorization_endpoint.as_deref(),
            p.resource.as_deref(),
            p.client_id.as_deref(),
            p.client_secret.as_deref(),
            p.access_token.as_deref(),
            p.refresh_token.as_deref(),
            p.expires_at,
            &p.token_type,
            p.scope.as_deref(),
            p.project_id.as_deref(),
            p.server_info,
            p.description.as_deref(),
            p.tags,
            p.catalog_slug.as_deref(),
            p.pkg.as_deref(),
        )
    }

    fn list_mcp_servers_with_status(&self, project_id: &str) -> anyhow::Result<Vec<Value>> {
        ModelService::list_mcp_servers_with_status(self, project_id)
    }

    fn count_mcp_tools(&self, project_id: &str) -> anyhow::Result<Value> {
        ModelService::count_mcp_tools(self, Some(project_id))
    }

    fn mcp_server_status(&self, server_id: &str, project_id: &str) -> anyhow::Result<Value> {
        ModelService::mcp_server_status(self, server_id, Some(project_id))
    }

    fn retry_mcp_server(&self, server_id: &str, project_id: &str) -> anyhow::Result<Value> {
        ModelService::retry_mcp_server(self, server_id, Some(project_id))
    }

    fn delete_mcp_server(&self, server_id: &str, project_id: &str) -> anyhow::Result<Value> {
        ModelService::delete_mcp_server(self, server_id, Some(project_id))
    }
}

/// Test seam for the blocking OAuth token exchange (Python tests patch
/// `MCPOAuthService`).
pub type ExchangeOAuthCodeFn =
    Arc<dyn Fn(TokenExchangeParams) -> Result<TokenResponse, MCPOAuthError> + Send + Sync>;
/// Test seam for remote discovery: `(server_url, access_token, token_type)`.
pub type DiscoverRemoteToolsFn = Arc<
    dyn Fn(&str, Option<&str>, &str) -> anyhow::Result<MCPConnectResult> + Send + Sync,
>;
/// Test seam for the Smartloop `/info` probe: receives the `/info` URL and
/// returns the parsed JSON.
pub type ProbeInfoFn = Arc<dyn Fn(&str) -> anyhow::Result<Value> + Send + Sync>;
/// Test seam replacing `MCPOAuthService()` (a fresh service per use).
pub type OAuthServiceFactory = Arc<dyn Fn() -> Box<dyn OAuthFlow> + Send + Sync>;
/// Test seam replacing `resolve_catalog_entry(slug)`.
pub type ResolveCatalogEntryFn = Arc<dyn Fn(&str) -> Option<MCPCatalogEntry> + Send + Sync>;

/// Overridable hooks replacing the network-touching collaborators.
#[derive(Clone, Default)]
pub struct McpRouteHooks {
    /// Replaces `_exchange_oauth_code` entirely (wins over `oauth_service`).
    pub exchange_oauth_code: Option<ExchangeOAuthCodeFn>,
    /// Replaces `_discover_remote_tools` (the `MCPClient` handshake).
    pub discover_remote_tools: Option<DiscoverRemoteToolsFn>,
    /// Replaces the `httpx.get(<url>/info)` probe.
    pub probe_info: Option<ProbeInfoFn>,
    /// Replaces `MCPOAuthService` for discovery, registration, the token
    /// exchange and the authorization URL.
    pub oauth_service: Option<OAuthServiceFactory>,
    /// Replaces `resolve_catalog_entry`.
    pub resolve_catalog_entry: Option<ResolveCatalogEntryFn>,
    /// Replaces the model service's MCP operations.
    pub mcp_registry: Option<Arc<dyn McpServerRegistry>>,
}

static HOOKS: Lazy<Mutex<McpRouteHooks>> = Lazy::new(|| Mutex::new(McpRouteHooks::default()));

/// Install (or with `McpRouteHooks::default()`, clear) the test hooks.
pub fn set_hooks(hooks: McpRouteHooks) {
    *HOOKS.lock() = hooks;
}

fn hooks() -> McpRouteHooks {
    HOOKS.lock().clone()
}

/// A fresh `MCPOAuthService()` (or the test stand-in).
fn open_oauth_service() -> Box<dyn OAuthFlow> {
    match hooks().oauth_service {
        Some(factory) => factory(),
        None => Box::new(ServiceOAuthFlow::new()),
    }
}

/// The MCP operations of *model_service* (or the test stand-in).
fn registry_for(model_service: &Arc<ModelService>) -> Arc<dyn McpServerRegistry> {
    match hooks().mcp_registry {
        Some(registry) => registry,
        None => model_service.clone(),
    }
}

/// The MCP operations of the global model service (`Depends(get_model_service)`).
fn default_registry() -> Arc<dyn McpServerRegistry> {
    match hooks().mcp_registry {
        Some(registry) => registry,
        None => get_model_service(),
    }
}

/// Resolve a catalog slug through the hook or the platform catalog.
fn lookup_catalog_entry(slug: &str) -> Option<MCPCatalogEntry> {
    match hooks().resolve_catalog_entry {
        Some(resolve) => resolve(slug),
        None => resolve_catalog_entry(slug),
    }
}

/// Query string of the GET OAuth callbacks.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, utoipa::IntoParams)]
#[into_params(parameter_in = Query)]
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

// ------------------------------------------------------------------ helpers

/// `time.time()`.
fn now_ts() -> f64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// `html.escape(s, quote=True)`.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            c => out.push(c),
        }
    }
    out
}

/// Python's `repr()` of a string (what `{x!r}` renders).
fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// `urllib.parse.quote_plus` (as used by `urlencode`).
fn quote_plus(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{:02X}", b)),
        }
    }
    out
}

/// Python truthiness of a JSON value (`None`/missing, `""`, `0`, `[]`, `{}`
/// and `false` are falsy).
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// `str(value)` of a non-null JSON value.
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// `session.get(key)` as an optional string (`None` for missing or null).
fn session_str(session: &OAuthSession, key: &str) -> Option<String> {
    match session.get(key) {
        None | Some(Value::Null) => None,
        Some(v) => Some(value_to_string(v)),
    }
}

/// `session.get(key) or None` — empty values read as absent.
fn session_truthy_str(session: &OAuthSession, key: &str) -> Option<String> {
    if truthy(session.get(key)) {
        session_str(session, key)
    } else {
        None
    }
}

/// `session[key]`: a missing key is Python's `KeyError`, whose `str()` is the
/// quoted key.
fn session_required(session: &OAuthSession, key: &str) -> Result<String, String> {
    match session.get(key) {
        None => Err(format!("'{key}'")),
        Some(Value::Null) => Ok(String::new()),
        Some(v) => Ok(value_to_string(v)),
    }
}

/// `session.get("tags")` as a list of strings.
fn session_tags(session: &OAuthSession) -> Option<Vec<String>> {
    match session.get("tags") {
        Some(Value::Array(items)) => Some(items.iter().map(value_to_string).collect()),
        _ => None,
    }
}

/// Python `x or None` for an optional string.
fn non_empty(s: Option<&str>) -> Option<String> {
    s.filter(|s| !s.is_empty()).map(str::to_string)
}

/// Schemes `urlunsplit` writes a `//netloc` for.
const USES_NETLOC: &[&str] = &[
    "", "ftp", "http", "gopher", "nntp", "telnet", "imap", "wais", "file", "mms", "https",
    "shttp", "snews", "prospero", "rtsp", "rtsps", "rtspu", "rsync", "svn", "svn+ssh", "sftp",
    "nfs", "git", "git+ssh", "ws", "wss", "itms-services",
];

/// Schemes whose last path segment carries `;params` (`urlparse`).
const USES_PARAMS: &[&str] = &[
    "", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu",
    "sip", "sips", "mms", "sftp", "tel",
];

/// `urllib.parse.urlsplit` result.
#[derive(Debug, Clone, Default, PartialEq)]
struct SplitUrl {
    scheme: String,
    netloc: String,
    path: String,
    query: String,
    fragment: String,
}

/// `urllib.parse.urlsplit`.
fn urlsplit(url: &str) -> SplitUrl {
    let mut rest = url;
    let mut scheme = String::new();
    if let Some(i) = url.find(':') {
        let candidate = &url[..i];
        let valid = candidate.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && candidate.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c));
        if valid {
            scheme = candidate.to_ascii_lowercase();
            rest = &url[i + 1..];
        }
    }
    let mut netloc = String::new();
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        netloc = after[..end].to_string();
        rest = &after[end..];
    }
    let (rest, fragment) = match rest.find('#') {
        Some(i) => (&rest[..i], rest[i + 1..].to_string()),
        None => (rest, String::new()),
    };
    let (path, query) = match rest.find('?') {
        Some(i) => (rest[..i].to_string(), rest[i + 1..].to_string()),
        None => (rest.to_string(), String::new()),
    };
    SplitUrl { scheme, netloc, path, query, fragment }
}

/// `urllib.parse.urlunsplit`.
fn urlunsplit(parts: &SplitUrl) -> String {
    let mut url = parts.path.clone();
    if !parts.netloc.is_empty()
        || (!parts.scheme.is_empty()
            && USES_NETLOC.contains(&parts.scheme.as_str())
            && !url.starts_with("//"))
    {
        if !url.is_empty() && !url.starts_with('/') {
            url = format!("/{url}");
        }
        url = format!("//{}{}", parts.netloc, url);
    }
    if !parts.scheme.is_empty() {
        url = format!("{}:{}", parts.scheme, url);
    }
    if !parts.query.is_empty() {
        url = format!("{}?{}", url, parts.query);
    }
    if !parts.fragment.is_empty() {
        url = format!("{}#{}", url, parts.fragment);
    }
    url
}

/// `urlparse(url).path`: the split path with any `;params` of its last
/// segment removed.
fn urlparse_path(parts: &SplitUrl) -> String {
    let path = &parts.path;
    if USES_PARAMS.contains(&parts.scheme.as_str()) && path.contains(';') {
        let cut = match path.rfind('/') {
            Some(slash) => path[slash..].find(';').map(|i| slash + i),
            None => path.find(';'),
        };
        if let Some(i) = cut {
            return path[..i].to_string();
        }
    }
    path.clone()
}

/// `parse_qs(query).get(key, [None])[0]`: the first non-blank value.
fn query_param_first(query: &str, key: &str) -> Option<String> {
    url::form_urlencoded::parse(query.as_bytes())
        .find(|(k, v)| k == key && !v.is_empty())
        .map(|(_, v)| v.into_owned())
}

/// `MCPDiscoveryError` (or its subclass `MCPAuthExpiredError`) anywhere in
/// the error chain.
fn is_discovery_error(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<MCPDiscoveryError>().is_some()
            || c.downcast_ref::<MCPAuthExpiredError>().is_some()
            || c.downcast_ref::<MCPClientError>().is_some_and(MCPClientError::is_discovery)
    })
}

/// A Python `ValueError` (or a pydantic validation error) in the error chain.
fn is_value_error(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        matches!(c.downcast_ref::<ModelServiceError>(), Some(ModelServiceError::Value(_)))
            || c.downcast_ref::<MCPDefinitionError>().is_some()
            || c.downcast_ref::<serde_json::Error>().is_some()
    })
}

/// The `MCPOAuthError` in the error chain, if any.
fn oauth_error(e: &anyhow::Error) -> Option<&MCPOAuthError> {
    e.chain().find_map(|c| c.downcast_ref::<MCPOAuthError>())
}

/// An exception the Python handler did not catch: FastAPI answers 500.
fn unhandled(context: &str, e: impl std::fmt::Display) -> HTTPException {
    tracing::error!(target: LOGGER, "{}: {}", context, e);
    HTTPException::new(500, "Internal Server Error")
}

/// `{"name", "description", "input_schema"}` per discovered tool.
fn tool_dicts(result: &MCPConnectResult) -> Vec<Value> {
    result
        .tools
        .iter()
        .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.input_schema}))
        .collect()
}

/// A service result as a dict (`_mcp_server_to_dict` output).
fn as_object(value: Value) -> Result<Map<String, Value>, String> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(format!("expected a dict, got {other}")),
    }
}

/// The pydantic type of one request-body field, for FastAPI's 422 errors.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FieldKind {
    /// `str` (required or with a non-null default): null is refused.
    Str,
    /// `Optional[str]`.
    OptStr,
    /// `Optional[List[str]]`.
    OptStrList,
    /// `Dict[str, str]`.
    StrMap,
}

/// One body field: `(name, kind, required)`.
type FieldSpec = (&'static str, FieldKind, bool);

/// `MCPRegisterRequest`'s fields.
const REGISTER_REQUEST_FIELDS: &[FieldSpec] = &[
    ("server_type", FieldKind::Str, false),
    ("name", FieldKind::OptStr, false),
    ("command", FieldKind::OptStr, false),
    ("args", FieldKind::OptStrList, false),
    ("env", FieldKind::StrMap, false),
    ("cwd", FieldKind::OptStr, false),
    ("server_url", FieldKind::OptStr, false),
    ("token", FieldKind::OptStr, false),
    ("callback_url", FieldKind::OptStr, false),
    ("return_url", FieldKind::OptStr, false),
    ("catalog_slug", FieldKind::OptStr, false),
];

/// `MCPDefinitionRequest`'s fields.
const DEFINITION_REQUEST_FIELDS: &[FieldSpec] =
    &[("toml", FieldKind::Str, true), ("filename", FieldKind::OptStr, false)];

/// `MCPDeviceStartRequest`'s fields.
const DEVICE_START_REQUEST_FIELDS: &[FieldSpec] = &[("catalog_slug", FieldKind::Str, true)];

/// `MCPDevicePollRequest`'s fields.
const DEVICE_POLL_REQUEST_FIELDS: &[FieldSpec] = &[("session_id", FieldKind::Str, true)];

/// `MCPOAuthCallbackRequest`'s fields.
const OAUTH_CALLBACK_REQUEST_FIELDS: &[FieldSpec] = &[
    ("server_url", FieldKind::Str, true),
    ("code", FieldKind::Str, true),
    ("state", FieldKind::Str, true),
    ("code_verifier", FieldKind::Str, true),
    ("client_id", FieldKind::Str, true),
    ("client_secret", FieldKind::OptStr, false),
    ("token_endpoint", FieldKind::Str, true),
    ("authorization_endpoint", FieldKind::Str, true),
    ("resource", FieldKind::OptStr, false),
    ("callback_url", FieldKind::OptStr, false),
];

/// One pydantic error entry.
fn validation_entry(kind: &str, loc: Vec<Value>, msg: &str, input: Value) -> Value {
    json!({"type": kind, "loc": loc, "msg": msg, "input": input})
}

/// FastAPI's `422 {"detail": [...]}` response.
fn validation_response(errors: Vec<Value>) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": errors }))).into_response()
}

/// Validate one field's value against its pydantic type, appending errors.
fn validate_field(name: &str, kind: FieldKind, value: &Value, errors: &mut Vec<Value>) {
    let string_error = |loc: Vec<Value>, input: &Value| {
        validation_entry("string_type", loc, "Input should be a valid string", input.clone())
    };
    match kind {
        FieldKind::Str => {
            if !value.is_string() {
                errors.push(string_error(vec![json!("body"), json!(name)], value));
            }
        }
        FieldKind::OptStr => {
            if !value.is_string() && !value.is_null() {
                errors.push(string_error(vec![json!("body"), json!(name)], value));
            }
        }
        FieldKind::OptStrList => match value {
            Value::Null => {}
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    if !item.is_string() {
                        errors.push(string_error(vec![json!("body"), json!(name), json!(i)], item));
                    }
                }
            }
            other => errors.push(validation_entry(
                "list_type",
                vec![json!("body"), json!(name)],
                "Input should be a valid list",
                other.clone(),
            )),
        },
        FieldKind::StrMap => match value {
            Value::Object(items) => {
                for (key, item) in items {
                    if !item.is_string() {
                        errors.push(string_error(vec![json!("body"), json!(name), json!(key)], item));
                    }
                }
            }
            other => errors.push(validation_entry(
                "dict_type",
                vec![json!("body"), json!(name)],
                "Input should be a valid dictionary",
                other.clone(),
            )),
        },
    }
}

/// Character offset of a serde_json error (Python's `JSONDecodeError.pos`).
fn json_error_pos(text: &str, e: &serde_json::Error) -> usize {
    let (line, column) = (e.line(), e.column());
    if line == 0 {
        return 0;
    }
    let preceding: usize = text.split('\n').take(line - 1).map(|l| l.chars().count() + 1).sum();
    preceding + column.saturating_sub(1)
}

/// Read and validate a JSON request body the way FastAPI does for a pydantic
/// body parameter, answering its `422` shape when it does not validate.
///
/// A missing `Content-Type` is read as JSON; a non-JSON one leaves the raw
/// bytes, which fail as "not a dictionary".  An empty body is a missing body.
fn parse_body<T: serde::de::DeserializeOwned>(
    headers: &HeaderMap,
    body: &[u8],
    fields: &[FieldSpec],
) -> Result<T, Response> {
    let is_json = match headers.get(CONTENT_TYPE).and_then(|v| v.to_str().ok()) {
        None => true,
        Some(content_type) => {
            let mime = content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
            match mime.split_once('/') {
                Some(("application", subtype)) => subtype == "json" || subtype.ends_with("+json"),
                _ => false,
            }
        }
    };
    if body.is_empty() {
        return Err(validation_response(vec![validation_entry(
            "missing",
            vec![json!("body")],
            "Field required",
            Value::Null,
        )]));
    }
    let text = String::from_utf8_lossy(body);
    let value = if is_json {
        match serde_json::from_slice::<Value>(body) {
            Ok(value) => value,
            Err(e) => {
                let mut entry = validation_entry(
                    "json_invalid",
                    vec![json!("body"), json!(json_error_pos(&text, &e))],
                    "JSON decode error",
                    json!({}),
                );
                entry["ctx"] = json!({ "error": e.to_string() });
                return Err(validation_response(vec![entry]));
            }
        }
    } else {
        Value::String(text.into_owned())
    };
    let object = match &value {
        Value::Object(object) => object,
        other => {
            return Err(validation_response(vec![validation_entry(
                "model_attributes_type",
                vec![json!("body")],
                "Input should be a valid dictionary or object to extract fields from",
                other.clone(),
            )]));
        }
    };
    let mut errors = Vec::new();
    for &(name, kind, required) in fields {
        match object.get(name) {
            None if required => errors.push(validation_entry(
                "missing",
                vec![json!("body"), json!(name)],
                "Field required",
                value.clone(),
            )),
            None => {}
            Some(field) => validate_field(name, kind, field, &mut errors),
        }
    }
    if !errors.is_empty() {
        return Err(validation_response(errors));
    }
    serde_json::from_value(value).map_err(|e| {
        validation_response(vec![validation_entry(
            "value_error",
            vec![json!("body")],
            &e.to_string(),
            Value::Null,
        )])
    })
}

// ------------------------------------------------------------------ state

/// Drop expired pending OAuth and device sessions and cached callback responses.
///
/// Piggybacked on request handling rather than run as a background task: the
/// maps only grow when a registration happens, so sweeping on the same events
/// keeps them bounded without any extra machinery.  A session without
/// `created_at` is stamped with *now* so it expires one TTL later.
pub fn purge_oauth_state(now: Option<f64>) {
    let now = now.unwrap_or_else(now_ts);
    {
        let mut sessions = PENDING_OAUTH_SESSIONS.lock();
        let mut expired = Vec::new();
        for (state, session) in sessions.iter_mut() {
            match session.get("created_at").and_then(Value::as_f64) {
                None => {
                    // Written by a path that predates the timestamp (or seeded
                    // by a test).  Stamp it here so it still expires one TTL
                    // from now, rather than reading as infinitely old and
                    // vanishing instantly.
                    session.insert("created_at".to_string(), json!(now));
                }
                Some(created) => {
                    if now - created > OAUTH_SESSION_TTL_S {
                        expired.push(state.clone());
                    }
                }
            }
        }
        for state in expired {
            tracing::info!(target: LOGGER, "Evicting expired OAuth session state={}", state);
            sessions.remove(&state);
        }
    }
    PENDING_DEVICE_SESSIONS.lock().retain(|_, session| now <= session.expires_at);
    let mut responses = COMPLETED_OAUTH_RESPONSES.lock();
    responses.retain(|_, (expires_at, _)| now <= *expires_at);
    while responses.len() > OAUTH_RESPONSE_MAX {
        responses.shift_remove_index(0);
    }
}

/// Run a blocking call on the single registration worker and await it.
///
/// Every MCP service/network call is synchronous — subprocess spawns with a
/// 2s startup poll, HTTP handshakes, local inference, an embedding pass.  Left
/// on the runtime they stall every other request (and every open SSE stream)
/// for the duration.  Errors come back through the return value unchanged, so
/// the callers' HTTPException mapping still works; panics are resumed.
pub async fn offload<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    let permit = EXECUTOR.acquire().await.expect("the MCP registration executor is never closed");
    // The permit travels with the job, so the worker stays busy until the call
    // finishes even if the awaiting request goes away.
    let joined = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        f()
    })
    .await;
    match joined {
        Ok(value) => value,
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        Err(e) => panic!("MCP registration worker failed: {e}"),
    }
}

/// Blocking OAuth token exchange, for use via [`offload`].
pub fn exchange_oauth_code(params: TokenExchangeParams) -> Result<TokenResponse, MCPOAuthError> {
    if let Some(exchange) = hooks().exchange_oauth_code {
        return exchange(params);
    }
    let mut oauth = open_oauth_service();
    oauth.exchange_code_for_tokens(&params)
}

/// Blocking remote MCP handshake + tool discovery, for use via [`offload`].
pub fn discover_remote_tools(
    server_url: &str,
    access_token: Option<&str>,
    token_type: &str,
) -> anyhow::Result<MCPConnectResult> {
    if let Some(discover) = hooks().discover_remote_tools {
        return discover(server_url, access_token, token_type);
    }
    let mut client = MCPClient::new();
    client.enter();
    let result = client.connect_and_discover_tools(server_url, access_token, token_type);
    client.exit();
    Ok(result)
}

/// Build a `server_info` dict from an `MCPServerInfo`, including description
/// and icons.
pub fn build_server_info_dict(si: Option<&MCPServerInfo>) -> Option<Map<String, Value>> {
    let si = si?;
    let mut d = Map::new();
    d.insert("name".to_string(), json!(si.name));
    d.insert("title".to_string(), json!(si.title));
    d.insert("version".to_string(), json!(si.version));
    if let Some(description) = si.description.as_deref().filter(|d| !d.is_empty()) {
        d.insert("description".to_string(), json!(description));
    }
    if let Some(icons) = si.icons.as_ref().filter(|i| !i.is_empty()) {
        d.insert("icons".to_string(), Value::Array(icons.clone()));
    }
    Some(d)
}

/// Build an `MCPServerResponse` from a `_mcp_server_to_dict` result.
///
/// `id`, `name`, `enabled` and `tools` are required: a missing one is the
/// Python `KeyError`, reported as its quoted key; a value of the wrong type is
/// the model's validation error.
pub fn build_server_response(result: &Map<String, Value>) -> Result<MCPServerResponse, String> {
    let required = |key: &str| result.get(key).cloned().ok_or_else(|| format!("'{key}'"));
    let optional = |key: &str| result.get(key).cloned().unwrap_or(Value::Null);
    let with_default = |key: &str, default: Value| result.get(key).cloned().unwrap_or(default);

    let id = required("id")?;
    let name = required("name")?;
    let enabled = required("enabled")?;
    let tools = required("tools")?;
    let body = json!({
        "id": id,
        "name": name,
        "description": optional("description"),
        "tags": with_default("tags", json!([])),
        "catalog_slug": optional("catalog_slug"),
        "command": optional("command"),
        "args": with_default("args", json!([])),
        "cwd": optional("cwd"),
        "server_type": with_default("server_type", json!("local")),
        "server_url": optional("server_url"),
        "enabled": enabled,
        "system": with_default("system", json!(false)),
        "tools": tools,
        "server_info": optional("server_info"),
        "status": with_default("status", json!("ready")),
        "last_error": optional("last_error"),
        "account": optional("account"),
    });
    serde_json::from_value(body).map_err(|e| e.to_string())
}

/// Render what the authorization server said it objected to (escaped HTML).
///
/// `error` alone comes from the fixed RFC 6749 §4.1.2.1 enum and so can only
/// ever name a category.  Which scope, which redirect URI, which client — that
/// lives in `error_description`, and dropping it is the difference between a
/// page the user can act on and the bare word "invalid_scope".
///
/// `error_uri` is attacker-controlled in the sense that matters here: it
/// arrives on a redirect from a server we discovered rather than one we
/// configured, so it is only ever rendered as a link when it is an http(s)
/// URL.  Anything else (`javascript:` above all) is shown as inert text.
pub fn authorization_error_detail(
    error: &str,
    error_description: Option<&str>,
    error_uri: Option<&str>,
) -> String {
    let mut detail = format!("<p>{}</p>", escape(error));
    if let Some(description) = error_description.filter(|d| !d.is_empty()) {
        detail.push_str(&format!("<p>{}</p>", escape(description)));
    }
    if let Some(uri) = error_uri.filter(|u| !u.is_empty()) {
        let lower = uri.to_lowercase();
        if lower.starts_with("https://") || lower.starts_with("http://") {
            detail.push_str(&format!(
                "<p><a href=\"{}\" rel=\"noopener noreferrer\">More information</a></p>",
                escape(uri)
            ));
        } else {
            detail.push_str(&format!("<p>{}</p>", escape(uri)));
        }
    }
    detail
}

/// Refuse a redirect that did not come from the server we sent the user to
/// (RFC 9207 mix-up defence).  Returns the 400 page to answer with, or `None`
/// to proceed.
///
/// Without it a client that talks to more than one authorization server can be
/// walked into a mix-up: the attacker starts a flow against a server it
/// controls, returns a code issued there to this callback, and the code is
/// redeemed at the *honest* server's token endpoint — which is the one named
/// in the session.  The `iss` parameter is what distinguishes the two.
///
/// Absence is checked as well as mismatch.  A server that advertises the
/// parameter and then omits it is either not the server it claims to be or
/// has had the parameter stripped in transit; treating a missing value as
/// "nothing to check" would hand the attacker the bypass for free.
///
/// Runs before anything else in the callback, so a response that fails it
/// costs nothing: the code is never presented to the token endpoint and the
/// pending session stays untouched for the real redirect.
pub fn reject_unexpected_issuer(state: Option<&str>, iss: Option<&str>) -> Option<HtmlPage> {
    let state = state.filter(|s| !s.is_empty())?;
    let (expected, advertised) = {
        let sessions = PENDING_OAUTH_SESSIONS.lock();
        // No session to compare against: the handler reports that as the
        // expired-session case it already is.
        let session = sessions.get(state).filter(|s| !s.is_empty())?;
        let expected = if truthy(session.get("issuer")) {
            session.get("issuer").map(value_to_string)
        } else {
            None
        };
        (expected, truthy(session.get("issuer_parameter_supported")))
    };
    let iss = iss.filter(|i| !i.is_empty());

    let detail = match (expected, iss) {
        (None, _) => {
            // RFC 8414 makes issuer mandatory, so this is a malformed metadata
            // document and there is nothing to compare against.  Refuse only
            // when the server also claimed it would identify itself, because
            // then it has contradicted itself; otherwise this is the
            // pre-RFC-9207 status quo.
            if !advertised {
                return None;
            }
            "it advertises the issuer parameter but published no issuer".to_string()
        }
        (Some(expected), Some(iss)) => {
            if iss == expected {
                return None;
            }
            format!("expected {}, got {}", py_repr(&expected), py_repr(iss))
        }
        (Some(_), None) => {
            if !advertised {
                // Publishes an issuer, does not send the parameter.  Common,
                // and not something the client can hold against it.
                return None;
            }
            "it advertises the issuer parameter but the response carried none".to_string()
        }
    };
    tracing::error!(
        target: LOGGER,
        "[mcp-oauth] rejecting authorization response for state={}: {}",
        state,
        detail
    );
    Some(HtmlPage::new(
        format!(
            "<html><body><h2>Authorization failed</h2>\
             <p>This response did not come from the authorization server the \
             connection was started with, so it was not used.</p>\
             <p>{}</p>\
             <p>You can close this window and try again.</p></body></html>",
            escape(&detail)
        ),
        400,
    ))
}

/// Reissue the authorization once, asking for no scopes at all (after an
/// `invalid_scope` refusal).  Returns the `302` redirect URL, or `None`.
///
/// A server can refuse the very set it published.  The PRM list is what RFC
/// 9728 §2 says to ask for, but it describes the resource rather than the
/// client, and a client registered through DCR may be permitted less than the
/// resource advertises.  There is nothing to negotiate against, so the one
/// useful retry is to name none and let the server apply the default for the
/// client it registered.
///
/// Exactly once.  `scope_retry` is what stops a server that refuses the empty
/// request too from bouncing the browser between here and the authorize
/// endpoint forever.  The session survives because this runs before the pop,
/// and the state is safe to reuse because a refused authorization issued no
/// code.
pub fn retry_authorization_without_scope(state: Option<&str>, error: &str) -> Option<String> {
    if error != "invalid_scope" {
        return None;
    }
    let state = state.filter(|s| !s.is_empty())?;
    let params = {
        let mut sessions = PENDING_OAUTH_SESSIONS.lock();
        let session = sessions.get_mut(state).filter(|s| !s.is_empty())?;
        if truthy(session.get("scope_retry")) {
            return None;
        }
        // Nothing to drop, or no challenge to rebuild the request around (a
        // session from a client driving the flow itself).
        if !truthy(session.get("scope")) || !truthy(session.get("code_challenge")) {
            return None;
        }
        tracing::info!(
            target: LOGGER,
            "[mcp-oauth] {} refused scope {} for state={}; retrying with none",
            session_str(session, "server_url").unwrap_or_else(|| "None".to_string()),
            py_repr(&session_str(session, "scope").unwrap_or_default()),
            state
        );
        session.insert("scope_retry".to_string(), Value::Bool(true));
        // Record what the second attempt actually asks for, so whatever the
        // grant comes back with is compared against the right request.
        session.insert("scope".to_string(), Value::Null);

        AuthorizationUrlParams {
            authorization_endpoint: session_str(session, "authorization_endpoint")
                .unwrap_or_default(),
            client_id: session_str(session, "client_id").unwrap_or_default(),
            redirect_uri: session_str(session, "callback_url").unwrap_or_default(),
            state: state.to_string(),
            code_challenge: session_str(session, "code_challenge").unwrap_or_default(),
            scope: None,
            resource: session_str(session, "resource"),
        }
    };
    let mut oauth = open_oauth_service();
    Some(oauth.build_authorization_url(&params))
}

/// Render the "connected" page shown in the browser after OAuth, with an
/// optional "Return to Smartloop" link carrying `server_id`.
///
/// It reports what was connected and stops there.  It used to navigate to the
/// app's `smartloop://` deep link automatically, which made the last step of
/// every install feel slow.  The app already learns about the server from the
/// API, so the page states the tool count and lets the user close the window;
/// a supplied return URL is offered as a link, so going back to the app stays
/// one click away without being forced.
pub fn oauth_success_page(
    server_name: &str,
    tool_count: usize,
    server_id: &str,
    return_url: Option<&str>,
) -> HtmlPage {
    let tools_label = if tool_count == 1 { "tool" } else { "tools" };
    let mut link = String::new();
    if let Some(return_url) = return_url.filter(|u| !u.is_empty()) {
        let separator = if return_url.contains('?') { "&" } else { "?" };
        let href = format!("{return_url}{separator}server_id={}", quote_plus(server_id));
        link = format!("<p><a href=\"{}\">Return to Smartloop</a></p>", escape(&href));
    }
    HtmlPage::ok(format!(
        "<html><head><title>MCP server connected</title></head>\
         <body style=\"font-family:system-ui,sans-serif;max-width:34rem;margin:4rem auto\">\
         <h2>MCP server connected</h2>\
         <p><strong>{}</strong> registered with {} {}.</p>\
         <p>Indexing and metadata finish in the background — you can close \
         this window.</p>\
         {}\
         <p><button onclick=\"window.close()\">Close window</button></p>\
         </body></html>",
        escape(server_name),
        tool_count,
        tools_label,
        link
    ))
}

// ------------------------------------------------------------ registration

/// Register a local (stdio) MCP server from a client-supplied command.
///
/// Standard MCP stdio config: the caller provides `command` (`uvx`, `npx`,
/// `python`, a path to a program, …) plus `args`; the command must be
/// installed on the host PATH.  No launcher is assumed — the command is run as
/// written.  When *curated* is a resolved catalog entry, its config and
/// metadata win over the request.
///
/// Returns as soon as the connection is persisted, with `status="pending"`:
/// launching the process, enumerating tools and generating metadata all
/// happen on the MCP background task.
pub async fn register_local_mcp_server(
    request: &MCPRegisterRequest,
    model_service: Arc<ModelService>,
    project_id: &str,
    curated: Option<&MCPCatalogEntry>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let local_cfg = curated.and_then(|c| c.local_config.as_ref());
    let command: Option<String> = match local_cfg {
        Some(cfg) => Some(cfg.command.clone()),
        None => request.command.clone(),
    };
    let args: Option<Vec<String>> = match local_cfg {
        Some(cfg) => Some(cfg.args.clone()),
        None => request.args.clone(),
    };
    let env: HashMap<String, String> = match local_cfg {
        Some(cfg) if !cfg.env.is_empty() => {
            cfg.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
        }
        _ => request.env.clone(),
    };

    let command = match command.filter(|c| !c.is_empty()) {
        Some(command) => command,
        None => {
            return Err(HTTPException::new(400, "'command' is required for local servers"));
        }
    };
    // A launcher (uvx, npx, python, …) with nothing to launch cannot start.  A
    // command that is itself a path to a program is complete without
    // arguments, so it is not held to this — the service applies the same rule.
    let has_args = args.as_ref().is_some_and(|a| !a.is_empty());
    let is_program_path =
        Path::new(&command).is_absolute() || command.contains(std::path::MAIN_SEPARATOR);
    if !has_args && !is_program_path {
        return Err(HTTPException::new(
            400,
            "'args' is required for local servers (e.g. args=['yahoo-finance-mcp'] \
             for uvx, args=['-y', '@scope/pkg'] for npx, or args=['-m', 'module'] \
             for python)",
        ));
    }

    let params = AddLocalMcpServer {
        command,
        args,
        name: match curated {
            Some(c) => Some(c.display_name.clone()),
            None => request.name.clone(),
        },
        env: Some(env),
        cwd: request.cwd.clone(),
        project_id: Some(project_id.to_string()),
        description: curated.and_then(|c| c.description.clone()),
        tags: curated.map(|c| c.tags.clone()),
        catalog_slug: non_empty(curated.map(|c| c.slug.as_str()))
            .or_else(|| request.catalog_slug.clone()),
    };
    let registry = registry_for(&model_service);
    let result = offload(move || registry.add_local_mcp_server(params)).await;
    let result = match result {
        Ok(result) => result,
        Err(e) if is_discovery_error(&e) => return Err(HTTPException::new(422, e.to_string())),
        Err(e) if is_value_error(&e) => return Err(HTTPException::new(400, e.to_string())),
        Err(e) => return Err(unhandled("Error registering local MCP server", e)),
    };
    let server = as_object(result)
        .and_then(|r| build_server_response(&r))
        .map_err(|e| unhandled("Error registering local MCP server", e))?;
    Ok(MCPRegisterResponse { auth_type: "none".to_string(), server: Some(server), ..Default::default() })
}

/// Blocking `httpx.get(info_url, timeout=10.0)` + `raise_for_status()` +
/// `json()`.  Redirects are not followed, as with httpx.
fn probe_info_default(info_url: &str) -> anyhow::Result<Value> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(reqwest::redirect::Policy::none())
        .build()?;
    let resp = client.get(info_url).send()?;
    let status = resp.status();
    if !status.is_success() {
        anyhow::bail!("HTTP {} for url '{}'", status, info_url);
    }
    Ok(resp.json::<Value>()?)
}

/// Try to register a Smartloop-hosted MCP server by probing its `/info`
/// endpoint.  Returns `Ok(None)` if the URL doesn't expose one, allowing the
/// caller to fall through to the standard MCP handshake.
pub async fn try_register_smartloop_mcp_server(
    server_url: &str,
    model_service: Arc<ModelService>,
    project_id: &str,
) -> Result<Option<MCPRegisterResponse>, HTTPException> {
    // Appending to the raw URL puts "/info" inside the query string when the
    // URL carries one ("...?token=abc/info"), which probes an endpoint that
    // does not exist and reads as "not a Smartloop server".  Split first so
    // the suffix lands on the path and the credential survives.
    let mut parts = urlsplit(server_url);
    parts.path = format!("{}/info", parts.path.trim_end_matches('/'));
    let info_url = urlunsplit(&parts);

    let probe = hooks().probe_info;
    let probed = offload(move || match probe {
        Some(probe) => probe(&info_url),
        None => probe_info_default(&info_url),
    })
    .await;
    let info = match probed {
        Ok(Value::Object(info)) if info.contains_key("pkg") => info,
        _ => return Ok(None),
    };

    let params = AddRemoteMcpServer {
        name: match info.get("name") {
            None => "smartloop-fetch".to_string(),
            Some(v) => v.as_str().unwrap_or("smartloop-fetch").to_string(),
        },
        server_url: server_url.to_string(),
        auth_type: "none".to_string(),
        tools: match info.get("tools") {
            None => Some(Vec::new()),
            Some(Value::Array(tools)) => Some(tools.clone()),
            Some(_) => None,
        },
        project_id: Some(project_id.to_string()),
        server_info: info.get("server_info").and_then(Value::as_object).cloned(),
        description: info.get("description").and_then(Value::as_str).map(str::to_string),
        pkg: info.get("pkg").and_then(|v| match v {
            Value::Null => None,
            other => Some(value_to_string(other)),
        }),
        ..Default::default()
    };
    let registry = registry_for(&model_service);
    let result = offload(move || registry.add_remote_mcp_server(params))
        .await
        .map_err(|e| unhandled("Error registering Smartloop MCP server", e))?;
    let server = as_object(result)
        .and_then(|r| build_server_response(&r))
        .map_err(|e| unhandled("Error registering Smartloop MCP server", e))?;
    Ok(Some(MCPRegisterResponse {
        auth_type: "none".to_string(),
        server: Some(server),
        ..Default::default()
    }))
}

/// Everything the OAuth discovery step hands back once its service is closed.
struct OAuthDiscovery {
    oauth_metadata: OAuthMetadata,
    registration: ClientRegistrationResponse,
    code_verifier: String,
    code_challenge: String,
    state: String,
    scope: Option<String>,
    resource: String,
    auth_url: String,
}

/// Why OAuth discovery stopped.
enum OAuthDiscoveryFailure {
    Http(HTTPException),
    OAuth(MCPOAuthError),
}

impl From<MCPOAuthError> for OAuthDiscoveryFailure {
    fn from(e: MCPOAuthError) -> Self {
        OAuthDiscoveryFailure::OAuth(e)
    }
}

/// Blocking OAuth discovery + dynamic client registration.
///
/// Every call in here is a synchronous HTTP round-trip to a third party, so
/// the whole sequence runs on the executor rather than the runtime.
fn oauth_discover(
    clean_url: &str,
    metadata_url: Option<&str>,
    callback_url: &str,
) -> Result<OAuthDiscovery, OAuthDiscoveryFailure> {
    let mut oauth = open_oauth_service();
    let mut oauth_metadata: Option<OAuthMetadata> = None;
    let mut prm: Option<ProtectedResourceMetadata> = None;
    match oauth.discover_protected_resource_metadata(clean_url, metadata_url) {
        Ok(found) => {
            let first_server = found.authorization_servers.first().cloned();
            prm = Some(found);
            if let Some(server) = first_server {
                // A failure here leaves the PRM in hand and falls back to the
                // server URL's own metadata below.
                oauth_metadata = oauth.discover_oauth_metadata(&server).ok();
            }
        }
        Err(_) => {}
    }
    let oauth_metadata = match oauth_metadata {
        Some(metadata) => metadata,
        None => oauth.discover_oauth_metadata(clean_url)?,
    };

    let registration_endpoint =
        match oauth_metadata.registration_endpoint.as_deref().filter(|e| !e.is_empty()) {
            Some(endpoint) => endpoint.to_string(),
            None => {
                return Err(OAuthDiscoveryFailure::Http(HTTPException::new(
                    400,
                    "Server requires authentication but does not support OAuth dynamic client registration",
                )));
            }
        };
    let registration = oauth.register_client(&registration_endpoint, callback_url)?;

    let (code_verifier, code_challenge) = oauth.generate_pkce();
    let state = oauth.generate_state();

    // RFC 9728 §2 defines the PRM list as the scope values "used in
    // authorization requests to request access to this protected resource" —
    // the one list that describes *this* server.  An authorization server's
    // RFC 8414 scopes_supported is a catalog of everything it knows across
    // every resource it fronts, so asking for all of it both overreaches
    // (§7.2: "as limited a scope as possible") and gets refused, since a
    // client registered through DCR is rarely permitted the whole catalog.
    //
    // A resource that declares nothing gets no scope parameter at all, which
    // leaves the authorization server to apply its own default for the
    // registered client.  That is the least-privilege answer.
    let scope = prm
        .as_ref()
        .filter(|p| !p.scopes_supported.is_empty())
        .map(|p| p.scopes_supported.join(" "));

    // RFC 8707 resource indicator, required by MCP auth (rev 2025-06-18).
    // Prefer the identifier the resource server publishes for itself; fall
    // back to the canonical server URL when it publishes no PRM.  Servers that
    // predate the requirement ignore the extra parameter.
    let resource = prm
        .as_ref()
        .and_then(|p| p.resource.clone())
        .filter(|r| !r.is_empty())
        .unwrap_or_else(|| oauth.canonical_resource(clean_url));

    let auth_url = oauth.build_authorization_url(&AuthorizationUrlParams {
        authorization_endpoint: oauth_metadata.authorization_endpoint.clone(),
        client_id: registration.client_id.clone(),
        redirect_uri: callback_url.to_string(),
        state: state.clone(),
        code_challenge: code_challenge.clone(),
        scope: scope.clone(),
        resource: Some(resource.clone()),
    });

    Ok(OAuthDiscovery {
        oauth_metadata,
        registration,
        code_verifier,
        code_challenge,
        state,
        scope,
        resource,
        auth_url,
    })
}

/// Labels of the required `credential_fields` on *curated* that *request*
/// leaves empty.  A token carried in the server URL's query satisfies a
/// "token" field.
fn missing_credentials(
    request: &MCPRegisterRequest,
    curated: Option<&MCPCatalogEntry>,
    has_token: bool,
) -> Vec<String> {
    let Some(rc) = curated.and_then(|c| c.remote_config.as_ref()) else {
        return Vec::new();
    };
    let supplied = serde_json::to_value(request).unwrap_or(Value::Null);
    rc.credential_fields
        .iter()
        .filter(|f| f.required)
        .filter(|f| {
            let given = supplied.get(&f.key).is_some_and(|v| match v {
                Value::String(s) => !s.is_empty(),
                v => !v.is_null(),
            });
            !(given || (f.key == "token" && has_token))
        })
        .map(|f| f.label.clone())
        .collect()
}

/// Register a remote MCP server.  Tries direct connection first, falls back
/// to OAuth (stashing a pending session and returning `auth_type="oauth"`).
///
/// When *curated* is a resolved catalog entry, its `server_url` and metadata
/// win over the request — the catalog is authoritative for servers we list, so
/// a stale client can't pin an outdated URL under a trusted name.
pub async fn register_remote_mcp_server(
    request: &MCPRegisterRequest,
    model_service: Arc<ModelService>,
    project_id: &str,
    curated: Option<&MCPCatalogEntry>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let curated_url = curated
        .and_then(|c| c.remote_config.as_ref())
        .map(|rc| rc.server_url.as_str());
    let effective_url = match non_empty(curated_url).or_else(|| non_empty(request.server_url.as_deref())) {
        Some(url) => url,
        None => {
            return Err(HTTPException::new(400, "'server_url' is required for remote servers"));
        }
    };

    let curated_description = curated.and_then(|c| c.description.clone());
    let curated_tags = curated.map(|c| c.tags.clone());
    // Provenance is the client's claim about where it installed from, and it
    // survives a lookup that missed.  `resolve_catalog_entry` never fails — a
    // stale cache, an expired platform token and an outage all read as "not
    // curated" — and dropping the slug there costs more than the metadata
    // does: the connection then pays for LLM enrichment it doesn't need.
    let curated_slug =
        non_empty(curated.map(|c| c.slug.as_str())).or_else(|| request.catalog_slug.clone());
    let curated_name = non_empty(curated.map(|c| c.display_name.as_str()));

    let server_url = effective_url.trim().to_string();

    // Probe for a Smartloop-hosted /info endpoint before attempting the MCP
    // handshake.  Any server that exposes /info with a "pkg" field is treated
    // as a Smartloop MCP server and registered directly — no SSE/OAuth
    // negotiation needed.
    if let Some(found) =
        try_register_smartloop_mcp_server(&server_url, model_service.clone(), project_id).await?
    {
        return Ok(found);
    }

    let parsed = urlsplit(&server_url);
    let token = non_empty(request.token.as_deref())
        .or_else(|| query_param_first(&parsed.query, "token"));
    let clean_url = if token.is_some() {
        format!("{}://{}{}", parsed.scheme, parsed.netloc, urlparse_path(&parsed))
    } else {
        server_url.clone()
    };

    // A curated entry that needs a pasted credential can't fall back to OAuth,
    // so name the missing credential up front.
    let missing = missing_credentials(request, curated, token.is_some());
    if !missing.is_empty() {
        return Err(HTTPException::new(
            400,
            format!("This connection needs {} to be supplied.", missing.join(" and ")),
        ));
    }

    let registry = registry_for(&model_service);

    // Try direct connection (no auth or token).
    let mut connect_result: Option<MCPConnectResult> = None;
    let direct = {
        let url = clean_url.clone();
        let tok = token.clone();
        offload(move || discover_remote_tools(&url, tok.as_deref(), "Bearer")).await
    };
    match direct {
        Err(_) => {
            tracing::debug!(target: LOGGER, "Direct connection failed for {}, trying OAuth", clean_url);
        }
        Ok(result) => {
            connect_result = Some(result.clone());
            if result.success {
                if result.tools.is_empty() {
                    // "MCP server has no tools available" — swallowed with
                    // every other direct-connect failure.
                    tracing::debug!(target: LOGGER, "Direct connection failed for {}, trying OAuth", clean_url);
                } else {
                    let si = result.server_info.as_ref();
                    let name = curated_name
                        .clone()
                        .or_else(|| non_empty(request.name.as_deref()))
                        .unwrap_or_else(|| match si {
                            Some(si) => si.name.clone().unwrap_or_default(),
                            None => parsed.netloc.clone(),
                        });
                    let params = AddRemoteMcpServer {
                        name,
                        server_url: clean_url.clone(),
                        auth_type: if token.is_some() { "token" } else { "none" }.to_string(),
                        token: token.clone(),
                        tools: Some(tool_dicts(&result)),
                        project_id: Some(project_id.to_string()),
                        server_info: build_server_info_dict(si),
                        description: curated_description
                            .clone()
                            .filter(|d| !d.is_empty())
                            .or_else(|| si.and_then(|s| s.description.clone())),
                        tags: curated_tags.clone(),
                        catalog_slug: curated_slug.clone(),
                        ..Default::default()
                    };
                    let reg = registry.clone();
                    let added = offload(move || reg.add_remote_mcp_server(params)).await;
                    match added
                        .map_err(|e| e.to_string())
                        .and_then(as_object)
                        .and_then(|r| build_server_response(&r))
                    {
                        Ok(server) => {
                            // Report what was stored so the client can tell a
                            // token connection from an open one.
                            return Ok(MCPRegisterResponse {
                                auth_type: if token.is_some() { "token" } else { "none" }.to_string(),
                                server: Some(server),
                                ..Default::default()
                            });
                        }
                        Err(_) => {
                            tracing::debug!(
                                target: LOGGER,
                                "Direct connection failed for {}, trying OAuth",
                                clean_url
                            );
                        }
                    }
                }
            }
        }
    }
    if let Some(result) = connect_result.as_ref().filter(|r| !r.success) {
        // A 401 here is the normal prelude to OAuth, but the detail also
        // carries the WWW-Authenticate header and body — the only clue when
        // the failure is a dead endpoint rather than a missing token.
        tracing::info!(
            target: LOGGER,
            "Direct connect to {} returned no usable session, falling back to OAuth: {}",
            clean_url,
            result.error.as_deref().unwrap_or("None")
        );
    }

    // A supplied token that fails to connect is reported as such; falling back
    // to OAuth would surface an unrelated DCR error instead.
    if token.is_some() {
        if connect_result.as_ref().is_some_and(|r| r.auth_expired) {
            return Err(HTTPException::new(
                401,
                format!(
                    "The access token was rejected by {}. \
                     Check it is correct and has not expired or been revoked.",
                    parsed.netloc
                ),
            ));
        }
        let detail = match connect_result.as_ref().and_then(|r| r.error.as_deref()) {
            Some(reason) => format!("Could not connect to {} with the access token: {reason}", parsed.netloc),
            None => format!("Could not connect to {} with the access token.", parsed.netloc),
        };
        return Err(HTTPException::new(502, detail));
    }

    // Fall back to OAuth.
    let callback_url = non_empty(request.callback_url.as_deref())
        .unwrap_or_else(|| default_callback_url(project_id));
    let metadata_url = connect_result.as_ref().and_then(|r| r.resource_metadata_url.clone());
    let discovered = {
        let url = clean_url.clone();
        let cb = callback_url.clone();
        offload(move || oauth_discover(&url, metadata_url.as_deref(), &cb)).await
    };
    let d = match discovered {
        Ok(d) => d,
        Err(OAuthDiscoveryFailure::Http(e)) => return Err(e),
        Err(OAuthDiscoveryFailure::OAuth(e)) => {
            return Err(HTTPException::new(
                400,
                format!("Failed to connect to MCP server and OAuth discovery failed: {e}"),
            ));
        }
    };

    // Store the session for the GET callback to use.
    purge_oauth_state(None);
    let session = json!({
        "created_at": now_ts(),
        "project_id": project_id,
        "server_url": clean_url,
        "state": d.state,
        "code_verifier": d.code_verifier,
        // The challenge is kept beside the verifier so an authorization that
        // has to be reissued — see the invalid_scope retry in the callback —
        // can be rebuilt without re-deriving it.
        "code_challenge": d.code_challenge,
        "scope": d.scope,
        "client_id": d.registration.client_id,
        "client_secret": d.registration.client_secret,
        "token_endpoint": d.oauth_metadata.token_endpoint,
        "authorization_endpoint": d.oauth_metadata.authorization_endpoint,
        // RFC 9207: who the redirect is expected to come back from, and
        // whether that server promised to say so.
        "issuer": d.oauth_metadata.issuer,
        "issuer_parameter_supported": d.oauth_metadata.authorization_response_iss_parameter_supported,
        "resource": d.resource,
        "callback_url": callback_url,
        "name": curated_name.or_else(|| non_empty(request.name.as_deref())),
        "return_url": request.return_url,
        // Curated metadata has to survive the redirect — the record is only
        // written in the callback, long after this request has returned.
        "description": curated_description,
        "tags": curated_tags,
        "catalog_slug": curated_slug,
    });
    if let Value::Object(session) = session {
        PENDING_OAUTH_SESSIONS.lock().insert(d.state.clone(), session);
    }

    Ok(MCPRegisterResponse {
        auth_type: "oauth".to_string(),
        server: None,
        auth_url: Some(d.auth_url),
        state: Some(d.state),
        code_verifier: Some(d.code_verifier),
        client_id: Some(d.registration.client_id),
        client_secret: d.registration.client_secret,
        token_endpoint: Some(d.oauth_metadata.token_endpoint),
        authorization_endpoint: Some(d.oauth_metadata.authorization_endpoint),
        resource: Some(d.resource),
        server_url: Some(clean_url),
    })
}

// ------------------------------------------------------------------ routes

/// Run a blocking registry call on the blocking pool.
async fn blocking<T, F>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    match tokio::task::spawn_blocking(f).await {
        Ok(value) => value,
        Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
        Err(e) => panic!("blocking MCP call failed: {e}"),
    }
}

/// Map a route's service error: `ValueError` → 404, anything else → 500 with
/// the message (logged).
fn route_error(context: &str, e: anyhow::Error) -> HTTPException {
    if is_value_error(&e) {
        return HTTPException::new(404, e.to_string());
    }
    tracing::error!(target: LOGGER, "{}: {:#}", context, e);
    HTTPException::new(500, e.to_string())
}

/// `GET /v1/projects/{project_id}/mcp` — `{"servers": [...]}` with each
/// server's `indexing` block, so a client can show "still connecting" without
/// a request per server.
#[utoipa::path(
    get,
    path = "/v1/projects/{project_id}/mcp",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID")),
    summary = "List MCP servers",
    description = "Each server includes its provisioning and indexing status.",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::MCPServerListResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn list_mcp_servers_for_project(
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<Value>, HTTPException> {
    let registry = default_registry();
    let servers = blocking(move || registry.list_mcp_servers_with_status(&project_id))
        .await
        .map_err(|e| route_error("Error listing MCP servers for project", e))?;
    Ok(Json(json!({ "servers": servers })))
}

/// `GET /v1/projects/{project_id}/mcp/tools/count` — diagnostic tool count.
///
/// `tools` is the total across registered servers as discovery recorded it —
/// the catalog a step is resolved against — and `by_server` breaks it down,
/// so a connection that discovered nothing is visible here rather than only in
/// the logs.
#[utoipa::path(
    get,
    path = "/v1/projects/{project_id}/mcp/tools/count",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID")),
    summary = "Count MCP tools",
    description = "Diagnostic: total tools across servers, with a per-server breakdown.",
    responses((status = 200, description = "OK", body = serde_json::Value), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn count_mcp_tools_for_project(
    AxumPath(project_id): AxumPath<String>,
) -> Result<Json<Value>, HTTPException> {
    let registry = default_registry();
    blocking(move || registry.count_mcp_tools(&project_id))
        .await
        .map(Json)
        .map_err(|e| route_error("Error counting MCP tools", e))
}

/// `GET /v1/projects/{project_id}/mcp/{server_id}` — one server with its
/// provisioning status.
///
/// Registration returns before the server is usable: tool discovery runs on
/// the MCP task.  `indexing.state` is what a client polls — `provisioning` →
/// `ready`, or `failed` with `indexing.error` — and `indexing.in_progress` is
/// the single flag to drive a spinner off.
#[utoipa::path(
    get,
    path = "/v1/projects/{project_id}/mcp/{server_id}",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID"), ("server_id" = String, Path, description = "MCP server ID")),
    summary = "Get an MCP server",
    description = "",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::MCPServerResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn get_mcp_server(
    AxumPath((project_id, server_id)): AxumPath<(String, String)>,
) -> Result<Json<Value>, HTTPException> {
    let registry = default_registry();
    blocking(move || registry.mcp_server_status(&server_id, &project_id))
        .await
        .map(Json)
        .map_err(|e| route_error("Error reading MCP server status", e))
}

/// `POST /v1/projects/{project_id}/mcp/{server_id}/refresh` — re-run
/// background provisioning for one server (retry a failed discovery).
///
/// Returns the server with `status="pending"`; the pass itself runs on the
/// MCP task.
#[utoipa::path(
    post,
    path = "/v1/projects/{project_id}/mcp/{server_id}/refresh",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID"), ("server_id" = String, Path, description = "MCP server ID")),
    summary = "Re-provision an MCP server",
    description = "",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::MCPServerResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn refresh_mcp_server(
    AxumPath((project_id, server_id)): AxumPath<(String, String)>,
) -> Result<Json<MCPServerResponse>, HTTPException> {
    let registry = default_registry();
    let result = blocking(move || registry.retry_mcp_server(&server_id, &project_id))
        .await
        .map_err(|e| route_error("Error refreshing MCP server", e))?;
    as_object(result)
        .and_then(|r| build_server_response(&r))
        .map(Json)
        .map_err(|e| {
            tracing::error!(target: LOGGER, "Error refreshing MCP server: {}", e);
            HTTPException::new(500, e)
        })
}

/// `DELETE /v1/projects/{project_id}/mcp/{server_id}`.
#[utoipa::path(
    delete,
    path = "/v1/projects/{project_id}/mcp/{server_id}",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID"), ("server_id" = String, Path, description = "MCP server ID")),
    summary = "Remove an MCP server",
    description = "",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::DeleteMCPServerResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn delete_mcp_server_from_project(
    AxumPath((project_id, server_id)): AxumPath<(String, String)>,
) -> Result<Json<DeleteMCPServerResponse>, HTTPException> {
    let registry = default_registry();
    let result = blocking(move || registry.delete_mcp_server(&server_id, &project_id))
        .await
        .map_err(|e| route_error("Error deleting MCP server from project", e))?;
    serde_json::from_value::<DeleteMCPServerResponse>(result).map(Json).map_err(|e| {
        tracing::error!(target: LOGGER, "Error deleting MCP server from project: {}", e);
        HTTPException::new(500, e.to_string())
    })
}

/// Unified MCP server registration (project-scoped).
///
/// For local servers, pass `server_type="local"` with `command` (e.g. "uvx",
/// "npx") and `args`.  For remote servers, pass `server_url` (`server_type`
/// defaults to "remote").  For Smartloop-hosted MCP servers, pass
/// `server_type="smartloop"` and `server_url`.
pub async fn register_mcp_server(
    project_id: &str,
    request: MCPRegisterRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPRegisterResponse, HTTPException> {
    // Resolve curated metadata server-side.  Dispatch on the *resolved*
    // entry's type rather than the caller's claim, so a future local catalog
    // listing works with no client release.  A miss leaves curated=None and
    // the install proceeds on the client-supplied values, enrichment and all.
    let curated = match non_empty(request.catalog_slug.as_deref()) {
        Some(slug) => blocking(move || lookup_catalog_entry(&slug)).await,
        None => None,
    };
    let server_type = match curated.as_ref() {
        Some(entry) => entry.server_type.clone(),
        None => Some(request.server_type.clone()),
    };

    if server_type.as_deref() == Some("local") {
        return register_local_mcp_server(&request, model_service, project_id, curated.as_ref())
            .await;
    }
    register_remote_mcp_server(&request, model_service, project_id, curated.as_ref()).await
}

/// Axum handler for `POST /v1/projects/{project_id}/mcp/register`.
#[utoipa::path(
    post,
    path = "/v1/projects/{project_id}/mcp/register",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID")),
    request_body = crate::api::model::mcp::MCPRegisterRequest,
    summary = "Register an MCP server",
    description = "Local (stdio) or remote (HTTP). A remote server that needs OAuth answers with an `auth_url` to complete sign-in.",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::MCPRegisterResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn register_mcp_server_handler(
    AxumPath(project_id): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: MCPRegisterRequest = match parse_body(&headers, &body, REGISTER_REQUEST_FIELDS) {
        Ok(request) => request,
        Err(response) => return response,
    };
    match register_mcp_server(&project_id, request, get_model_service()).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => e.into_response(),
    }
}

/// Install a connection from a dropped-in TOML file.
///
/// The file is the whole definition: what the server is called and how to
/// reach it.  It is kept — saved under `~/.smartloop/mcp.d/` so the connection
/// can be re-created, edited or removed as a file — and the connection itself
/// goes through the ordinary registration path, so a remote server needing
/// OAuth returns an `auth_url` exactly as it would if the URL had been typed.
pub async fn register_mcp_from_definition(
    project_id: &str,
    request: MCPDefinitionRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPRegisterResponse, HTTPException> {
    let home_dir = AppSettings::new().home_dir();
    register_mcp_from_definition_with(project_id, request, Path::new(&home_dir), |pid, req| {
        let model_service = model_service.clone();
        async move { register_mcp_server(&pid, req, model_service).await }
    })
    .await
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
    let source = non_empty(request.filename.as_deref()).unwrap_or_else(|| "upload".to_string());
    let found = parse_definitions(&request.toml, &source).map_err(|e| {
        HTTPException::new(400, format!("Could not read this connection file: {e}"))
    })?;
    if found.is_empty() {
        return Err(HTTPException::new(
            400,
            "No [[mcp]] entry in this file — it needs at least a name \
             and either a local command or a remote url.",
        ));
    }
    if found.len() > 1 {
        let mut keys: Vec<&str> = found.keys().map(String::as_str).collect();
        keys.sort_unstable();
        return Err(HTTPException::new(
            400,
            format!(
                "This file describes {} connections ({}); drop one file per connection.",
                found.len(),
                keys.join(", ")
            ),
        ));
    }

    let definition = found.into_values().next().expect("exactly one definition");
    let kind = match definition.kind() {
        Some(kind) => kind,
        None => {
            return Err(HTTPException::new(
                400,
                format!(
                    "'{}' has no connection: add a url for a remote server, or a package \
                     name for one to run locally.",
                    definition.name
                ),
            ));
        }
    };

    // Persist before registering, so a file dropped in is on disk even if the
    // registration below fails and has to be retried.
    let key = definition.key();
    let saved = (|| -> std::io::Result<std::path::PathBuf> {
        let drop_in_dir = home_dir.join(USER_DEFINITIONS_DIR);
        std::fs::create_dir_all(&drop_in_dir)?;
        // Named after the connection, not after whatever the file was called
        // on the way in — that is what makes re-dropping an edited file
        // replace the old definition instead of stacking a second copy.
        let target = drop_in_dir.join(format!("{key}.toml"));
        std::fs::write(&target, request.toml.as_bytes())?;
        Ok(target)
    })();
    match saved {
        Ok(target) => {
            reload_definitions();
            tracing::info!(
                target: LOGGER,
                "[mcp-definition] Saved '{}' to {}",
                key,
                target.display()
            );
        }
        Err(e) => {
            // The connection is still worth installing without the file kept.
            tracing::warn!(target: LOGGER, "[mcp-definition] Could not save the definition file: {}", e);
        }
    }

    let (command, args) = definition.launch();
    let register_request = MCPRegisterRequest {
        name: Some(definition.name.clone()),
        server_type: kind.to_string(),
        command,
        args: Some(args),
        env: definition.env.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        cwd: definition.cwd.clone(),
        server_url: definition.url.clone(),
        token: definition.token.clone(),
        callback_url: None,
        return_url: Some("smartloop://mcp-authorization".to_string()),
        catalog_slug: None,
    };
    register(project_id.to_string(), register_request).await
}

/// Axum handler for `POST /v1/projects/{project_id}/mcp/definition`.
#[utoipa::path(
    post,
    path = "/v1/projects/{project_id}/mcp/definition",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID")),
    request_body = crate::api::model::mcp::MCPDefinitionRequest,
    summary = "Register an MCP server from a TOML definition",
    description = "",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::MCPRegisterResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
pub async fn register_mcp_from_definition_handler(
    AxumPath(project_id): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: MCPDefinitionRequest = match parse_body(&headers, &body, DEFINITION_REQUEST_FIELDS) {
        Ok(request) => request,
        Err(response) => return response,
    };
    match register_mcp_from_definition(&project_id, request, get_model_service()).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => e.into_response(),
    }
}

/// `GET /v1/projects/{project_id}/mcp/oauth/callback` — where a provider's
/// redirect lands, scoped to the project it belongs to.
///
/// The project in the path is what gets this request forwarded to the child
/// that owns it, which is both where the pending session was stored and the
/// only process whose `save()` writes that project's file.  See
/// [`default_callback_url`].
pub async fn mcp_oauth_callback_for_project(
    project_id: &str,
    query: OAuthCallbackQuery,
    model_service: Arc<ModelService>,
) -> OAuthCallbackResponse {
    complete_oauth_redirect(query, model_service, Some(project_id)).await
}

/// `GET /v1/mcp/oauth/callback` — the unscoped redirect target, kept for a
/// client that asks for it.
///
/// Nothing generates this URL any more — [`default_callback_url`] scopes it to
/// the project — but a caller may still pass it as `callback_url`, and an
/// unsupervised server can serve it correctly because it owns every project.
/// Under supervision it cannot, and says so rather than reporting success for
/// a connection it dropped.
pub async fn mcp_oauth_callback(
    query: OAuthCallbackQuery,
    model_service: Arc<ModelService>,
) -> OAuthCallbackResponse {
    complete_oauth_redirect(query, model_service, None).await
}

#[utoipa::path(
    get,
    path = "/v1/projects/{project_id}/mcp/oauth/callback",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID"), OAuthCallbackQuery),
    summary = "MCP OAuth redirect target (project)",
    description = "Browser redirect target of an MCP server's OAuth flow.",
    responses((status = 200, description = "OAuth completion page", content_type = "text/html"), (status = 302, description = "Redirect back to the requesting app"))
)]
async fn mcp_oauth_callback_for_project_handler(
    AxumPath(project_id): AxumPath<String>,
    Query(query): Query<OAuthCallbackQuery>,
) -> OAuthCallbackResponse {
    mcp_oauth_callback_for_project(&project_id, query, get_model_service()).await
}

#[utoipa::path(
    get,
    path = "/v1/mcp/oauth/callback",
    tag = "mcp",
    params(OAuthCallbackQuery),
    summary = "MCP OAuth redirect target",
    description = "Browser redirect target of an MCP server's OAuth flow.",
    responses((status = 200, description = "OAuth completion page", content_type = "text/html"), (status = 302, description = "Redirect back to the requesting app"))
)]
async fn mcp_oauth_callback_handler(Query(query): Query<OAuthCallbackQuery>) -> OAuthCallbackResponse {
    mcp_oauth_callback(query, get_model_service()).await
}

/// Why completing the redirect failed after the session was taken.
enum CallbackFailure {
    OAuth(MCPOAuthError),
    Other(String),
}

impl From<MCPOAuthError> for CallbackFailure {
    fn from(e: MCPOAuthError) -> Self {
        CallbackFailure::OAuth(e)
    }
}

impl CallbackFailure {
    fn from_anyhow(e: anyhow::Error) -> Self {
        match oauth_error(&e) {
            Some(oauth) => CallbackFailure::OAuth(oauth.clone()),
            None => CallbackFailure::Other(e.to_string()),
        }
    }
}

/// Exchange the code for tokens, discover tools, and register the server.
pub async fn complete_oauth_redirect(
    query: OAuthCallbackQuery,
    model_service: Arc<ModelService>,
    routed_project_id: Option<&str>,
) -> OAuthCallbackResponse {
    let OAuthCallbackQuery { code, state, iss, error, error_description, error_uri } = query;
    let code = code.filter(|c| !c.is_empty());
    let state_ref = state.as_deref().filter(|s| !s.is_empty());

    // Before anything reads this response — including the invalid_scope
    // retry, which would otherwise reissue an authorization on an attacker's
    // say-so.
    if let Some(page) = reject_unexpected_issuer(state_ref, iss.as_deref()) {
        return page.into();
    }

    if let Some(error) = error.as_deref().filter(|e| !e.is_empty()) {
        // RFC 6749 §4.1.2.1 sends three fields and only `error` is from a
        // fixed enum.  `error_description` is the sentence that says *which*
        // scope, or *which* redirect URI, the server objected to.
        tracing::warn!(
            target: LOGGER,
            "[mcp-oauth] provider refused authorization for state={}: error={} description={} uri={}",
            state.as_deref().unwrap_or("None"),
            error,
            error_description.as_deref().unwrap_or("None"),
            error_uri.as_deref().unwrap_or("None")
        );
        if let Some(url) = retry_authorization_without_scope(state_ref, error) {
            return OAuthCallbackResponse::Redirect(url);
        }
        return HtmlPage::new(
            format!(
                "<html><body><h2>Authorization failed</h2>{}<p>You can close this window.</p></body></html>",
                authorization_error_detail(error, error_description.as_deref(), error_uri.as_deref())
            ),
            400,
        )
        .into();
    }

    let (code, state) = match (code, state_ref) {
        (Some(code), Some(state)) => (code, state.to_string()),
        (code, state) => {
            tracing::warn!(
                target: LOGGER,
                "OAuth callback missing parameters: code={}, state={}",
                if code.is_some() { "True" } else { "False" },
                if state.is_some() { "True" } else { "False" }
            );
            return HtmlPage::new(
                "<html><body><h2>Missing parameters</h2>\
                 <p>No authorization code or state received.</p></body></html>",
                400,
            )
            .into();
        }
    };

    tracing::debug!(target: LOGGER, "OAuth callback received: state={}", state);

    // Return the cached response for duplicate browser requests.
    purge_oauth_state(None);
    if let Some((_, cached)) = COMPLETED_OAUTH_RESPONSES.lock().get(&state).cloned() {
        tracing::info!(target: LOGGER, "Returning cached OAuth response for state={}", state);
        return cached.into();
    }

    let session = PENDING_OAUTH_SESSIONS.lock().remove(&state).filter(|s| !s.is_empty());
    let session = match session {
        Some(session) => session,
        None => {
            tracing::warn!(target: LOGGER, "OAuth session not found for state={}", state);
            return HtmlPage::new(
                "<html><body><h2>Session expired</h2>\
                 <p>OAuth session not found. Please try again.</p></body></html>",
                400,
            )
            .into();
        }
    };

    let project_id = session_str(&session, "project_id");
    if let Some(routed) = routed_project_id {
        if Some(routed) != project_id.as_deref() {
            // The redirect was addressed to one project and the session
            // belongs to another: this is the wrong child, and its workspace
            // has no such project to save into.
            tracing::error!(
                target: LOGGER,
                "[mcp-oauth] callback routed to project {} but state={} belongs to {}",
                routed,
                state,
                project_id.as_deref().unwrap_or("None")
            );
            return wrong_process_page(Some(routed)).into();
        }
    }
    if !can_persist_project(project_id.as_deref()) {
        tracing::error!(
            target: LOGGER,
            "[mcp-oauth] callback for project {} reached a process that cannot write it; \
             refusing before the code is spent",
            project_id.as_deref().unwrap_or("None")
        );
        return wrong_process_page(project_id.as_deref()).into();
    }

    match finish_oauth_redirect(&session, &state, code, model_service).await {
        Ok(response) => response,
        Err(CallbackFailure::OAuth(e)) => {
            tracing::error!(target: LOGGER, "MCP OAuth callback error: {}", e);
            HtmlPage::new(
                format!(
                    "<html><body><h2>Authorization failed</h2><p>{e}</p>\
                     <p>You can close this window and try again.</p></body></html>"
                ),
                400,
            )
            .into()
        }
        Err(CallbackFailure::Other(e)) => {
            tracing::error!(target: LOGGER, "MCP OAuth callback error: {}", e);
            HtmlPage::new(
                format!(
                    "<html><body><h2>Error</h2><p>{e}</p>\
                     <p>You can close this window and try again.</p></body></html>"
                ),
                500,
            )
            .into()
        }
    }
}

/// The `try` body of `_complete_oauth_redirect`, once the session is taken.
async fn finish_oauth_redirect(
    session: &OAuthSession,
    state: &str,
    code: String,
    model_service: Arc<ModelService>,
) -> Result<OAuthCallbackResponse, CallbackFailure> {
    let req = |key: &str| session_required(session, key).map_err(CallbackFailure::Other);

    // Exchange the code for tokens.
    let params = TokenExchangeParams {
        token_endpoint: req("token_endpoint")?,
        code,
        redirect_uri: req("callback_url")?,
        client_id: req("client_id")?,
        code_verifier: req("code_verifier")?,
        client_secret: session_str(session, "client_secret"),
        resource: session_str(session, "resource"),
    };
    let token_response = offload(move || exchange_oauth_code(params)).await?;

    let expires_at = token_response
        .expires_in
        .filter(|e| *e != 0)
        .map(|e| now_ts() + e as f64);

    // Discover tools.
    let server_url = req("server_url")?;
    let connect_result = {
        let url = server_url.clone();
        let access_token = token_response.access_token.clone();
        let token_type = token_response.token_type.clone();
        offload(move || discover_remote_tools(&url, Some(&access_token), &token_type))
            .await
            .map_err(CallbackFailure::from_anyhow)?
    };

    let tools = if connect_result.success { tool_dicts(&connect_result) } else { Vec::new() };
    if tools.is_empty() {
        // Two very different failures reach this point — the handshake was
        // refused, or it succeeded and the server offered nothing.  Collapsing
        // both into "no tools available" hides the reason (a dead transport, a
        // rejected token, a grant with no product scopes), so name it.  The
        // grant is right here, so report what it came back with rather than
        // guessing that scopes were the problem.
        let reason = if let Some(error) = connect_result.error.as_deref().filter(|e| !e.is_empty()) {
            error.to_string()
        } else if let Some(scope) = token_response.scope.as_deref().filter(|s| !s.is_empty()) {
            format!(
                "The server completed the MCP handshake but exposes no tools. \
                 The authorization granted: {scope}."
            )
        } else {
            "The server completed the MCP handshake but exposes no tools, and the \
             authorization came back naming no scopes."
                .to_string()
        };
        tracing::error!(
            target: LOGGER,
            "Tool discovery failed after OAuth for {}: {}",
            server_url,
            reason
        );
        return Ok(HtmlPage::new(
            format!(
                "<html><body><h2>Registration failed</h2>\
                 <p>Could not list tools on {}.</p><pre>{}</pre></body></html>",
                escape(&server_url),
                escape(&reason)
            ),
            400,
        )
        .into());
    }

    let si = if connect_result.success { connect_result.server_info.as_ref() } else { None };
    let server_name = session_truthy_str(session, "name")
        .or_else(|| si.and_then(|s| non_empty(s.name.as_deref())))
        .unwrap_or_else(|| "Remote MCP Server".to_string());

    // Register the server.
    let params = AddRemoteMcpServer {
        name: server_name,
        server_url: server_url.clone(),
        auth_type: "oauth".to_string(),
        tools: Some(tools),
        token_endpoint: Some(req("token_endpoint")?),
        authorization_endpoint: Some(req("authorization_endpoint")?),
        resource: session_str(session, "resource"),
        client_id: Some(req("client_id")?),
        client_secret: session_str(session, "client_secret"),
        access_token: Some(token_response.access_token.clone()),
        refresh_token: token_response.refresh_token.clone(),
        expires_at,
        token_type: token_response.token_type.clone(),
        scope: token_response.scope.clone(),
        project_id: Some(req("project_id")?),
        server_info: build_server_info_dict(si),
        // Curated copy stashed at register time wins over whatever the vendor
        // advertises in its handshake.
        description: session_truthy_str(session, "description")
            .or_else(|| si.and_then(|s| s.description.clone())),
        tags: session_tags(session),
        catalog_slug: session_str(session, "catalog_slug"),
        token: None,
        pkg: None,
    };
    let registry = registry_for(&model_service);
    let result = offload(move || registry.add_remote_mcp_server(params))
        .await
        .map_err(CallbackFailure::from_anyhow)?;
    let result = as_object(result).map_err(CallbackFailure::Other)?;

    let server_id = result.get("id").map(value_to_string).ok_or_else(|| CallbackFailure::Other("'id'".to_string()))?;
    let tool_count = match result.get("tools") {
        None => 0,
        Some(Value::Array(items)) => items.len(),
        Some(Value::Object(items)) => items.len(),
        Some(Value::String(s)) => s.chars().count(),
        Some(other) => {
            return Err(CallbackFailure::Other(format!("object of type '{other}' has no len()")));
        }
    };
    let name = result
        .get("name")
        .map(|v| match v {
            Value::Null => "None".to_string(),
            other => value_to_string(other),
        })
        .ok_or_else(|| CallbackFailure::Other("'name'".to_string()))?;
    let return_url = session_str(session, "return_url");
    let page = oauth_success_page(&name, tool_count, &server_id, return_url.as_deref());

    // Cache the response for duplicate browser requests.
    COMPLETED_OAUTH_RESPONSES
        .lock()
        .insert(state.to_string(), (now_ts() + OAUTH_RESPONSE_TTL_S, page.clone()));
    Ok(page.into())
}

// ---------------------------------------------------------- OAuth flow

/// Where the provider sends the browser back when the client names nowhere:
/// the supervisor's public port, scoped to the project.
///
/// It is addressed to the **supervisor's** public port rather than whatever
/// port served the registration.  Under supervision that was the child's
/// ephemeral one, so the redirect URI registered with the provider named a
/// port that disappears when the child is idle-reaped.
///
/// And it is **scoped to the project**, so the supervisor forwards the
/// redirect to the child that owns it (see
/// `crate::supervisor::routing::project_to_proxy`).  The unscoped path is the
/// one route here with no project in it, which left it answered by the
/// supervisor: the pending session was in the child's memory, and a
/// supervisor that has delegated its projects writes no `project.json` at all.
pub fn default_callback_url(project_id: &str) -> String {
    format!(
        "http://localhost:{}/v1/projects/{}/mcp/oauth/callback",
        AppSettings::new().api_port,
        project_id
    )
}

/// Whether a server saved here would reach *project_id*'s own file.
///
/// `Workspace::save()` writes a project's file only for the process that owns
/// the project, and says nothing when it declines — so a callback answered in
/// the wrong process returns a success page for a connection that was never
/// written.  Checked before the code is exchanged, because an authorization
/// code is single-use: burning it and then failing to persist leaves the user
/// with no way forward but to start again.
pub fn can_persist_project(project_id: Option<&str>) -> bool {
    let project_id = match project_id.filter(|p| !p.is_empty()) {
        Some(project_id) => project_id,
        None => return false,
    };
    if let Some(pinned) = pinned_project_id() {
        return pinned == project_id;
    }
    if projects_delegated() {
        return owned_project_id().as_deref() == Some(project_id);
    }
    // Unpinned and undelegated — the --no-supervised server or the CLI, which
    // is the only process there is and writes every project's file.
    true
}

/// The 500 page for a callback answered by a process that cannot save the
/// project.
pub fn wrong_process_page(project_id: Option<&str>) -> HtmlPage {
    HtmlPage::new(
        format!(
            "<html><body><h2>Registration failed</h2>\
             <p>This authorization was completed by a part of Smartloop that \
             cannot save connections for project {}.</p>\
             <p>You can close this window and add the connection again.</p>\
             </body></html>",
            escape(project_id.unwrap_or("None"))
        ),
        500,
    )
}

/// `POST /v1/projects/{project_id}/mcp/oauth/callback` — exchange an OAuth
/// authorization code for tokens and register the remote MCP server.
pub async fn complete_mcp_oauth(
    project_id: &str,
    request: MCPOAuthCallbackRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPServerResponse, HTTPException> {
    let callback_url = non_empty(request.callback_url.as_deref())
        .unwrap_or_else(|| "http://localhost:9876/callback".to_string());

    // Exchange the code for tokens.
    let params = TokenExchangeParams {
        token_endpoint: request.token_endpoint.clone(),
        code: request.code.clone(),
        redirect_uri: callback_url,
        client_id: request.client_id.clone(),
        code_verifier: request.code_verifier.clone(),
        client_secret: request.client_secret.clone(),
        resource: request.resource.clone(),
    };
    let token_response = offload(move || exchange_oauth_code(params))
        .await
        .map_err(|e| HTTPException::new(400, format!("OAuth error: {e}")))?;

    let expires_at = token_response
        .expires_in
        .filter(|e| *e != 0)
        .map(|e| now_ts() + e as f64);

    let map_err = |e: anyhow::Error| -> HTTPException {
        if let Some(oauth) = oauth_error(&e) {
            return HTTPException::new(400, format!("OAuth error: {oauth}"));
        }
        if is_value_error(&e) {
            return HTTPException::new(404, e.to_string());
        }
        tracing::error!(target: LOGGER, "Error completing MCP OAuth: {:#}", e);
        HTTPException::new(500, e.to_string())
    };

    // Discover tools.
    let connect_result = {
        let url = request.server_url.clone();
        let access_token = token_response.access_token.clone();
        let token_type = token_response.token_type.clone();
        offload(move || discover_remote_tools(&url, Some(&access_token), &token_type))
            .await
            .map_err(map_err)?
    };

    let tools = if connect_result.success { tool_dicts(&connect_result) } else { Vec::new() };
    let si = if connect_result.success { connect_result.server_info.as_ref() } else { None };
    let server_info_dict = build_server_info_dict(si);

    // The curated copy stashed at register time, for a client that posts the
    // grant back instead of letting the browser redirect land on the GET
    // callback.  Both completions write the same record, so both take their
    // name, description, tags and provenance from the same place — read
    // rather than popped, since the redirect may still be in flight.
    let session: OAuthSession =
        PENDING_OAUTH_SESSIONS.lock().get(&request.state).cloned().unwrap_or_default();

    let name = session_truthy_str(&session, "name").unwrap_or_else(|| match si {
        Some(si) => si.name.clone().unwrap_or_default(),
        None => "Remote MCP Server".to_string(),
    });
    let params = AddRemoteMcpServer {
        name,
        server_url: request.server_url.clone(),
        auth_type: "oauth".to_string(),
        tools: Some(tools),
        token_endpoint: Some(request.token_endpoint.clone()),
        authorization_endpoint: Some(request.authorization_endpoint.clone()),
        resource: request.resource.clone(),
        client_id: Some(request.client_id.clone()),
        client_secret: request.client_secret.clone(),
        access_token: Some(token_response.access_token.clone()),
        refresh_token: token_response.refresh_token.clone(),
        expires_at,
        token_type: token_response.token_type.clone(),
        scope: token_response.scope.clone(),
        project_id: Some(project_id.to_string()),
        server_info: server_info_dict,
        description: session_truthy_str(&session, "description")
            .or_else(|| si.and_then(|s| s.description.clone())),
        tags: session_tags(&session),
        catalog_slug: session_str(&session, "catalog_slug"),
        token: None,
        pkg: None,
    };
    let registry = registry_for(&model_service);
    let result = offload(move || registry.add_remote_mcp_server(params)).await.map_err(map_err)?;

    as_object(result).and_then(|r| build_server_response(&r)).map_err(|e| {
        tracing::error!(target: LOGGER, "Error completing MCP OAuth: {}", e);
        HTTPException::new(500, e)
    })
}

#[utoipa::path(
    post,
    path = "/v1/projects/{project_id}/mcp/oauth/callback",
    tag = "mcp",
    params(("project_id" = String, Path, description = "Project ID")),
    request_body = crate::api::model::mcp::MCPOAuthCallbackRequest,
    summary = "Complete MCP OAuth sign-in",
    description = "",
    responses((status = 200, description = "OK", body = crate::api::model::mcp::MCPServerResponse), (status = 400, description = "Invalid request"), (status = 404, description = "Not found"), (status = 500, description = "Server error"))
)]
async fn complete_mcp_oauth_handler(
    AxumPath(project_id): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: MCPOAuthCallbackRequest = match parse_body(&headers, &body, OAUTH_CALLBACK_REQUEST_FIELDS) {
        Ok(request) => request,
        Err(response) => return response,
    };
    match complete_mcp_oauth(&project_id, request, get_model_service()).await {
        Ok(server) => Json(server).into_response(),
        Err(e) => e.into_response(),
    }
}

// ------------------------------------------------------------------ device sign-in

/// Start an RFC 8628 device sign-in for a catalog entry that has `device_flow`.
///
/// The provider's device code stays in [`PENDING_DEVICE_SESSIONS`]; the client
/// gets the code to show the user and a session id to poll with.
pub async fn start_device_sign_in(
    project_id: &str,
    request: MCPDeviceStartRequest,
) -> Result<MCPDeviceStartResponse, HTTPException> {
    let slug = request.catalog_slug.clone();
    let remote = blocking(move || lookup_catalog_entry(&slug)).await.and_then(|e| e.remote_config);
    let Some((server_url, flow)) = remote.and_then(|rc| Some((rc.server_url, rc.device_flow?))) else {
        return Err(HTTPException::new(
            400,
            format!("'{}' does not offer a device sign-in", request.catalog_slug),
        ));
    };

    let started = {
        let flow = flow.clone();
        let scope = flow.scopes.join(" ");
        blocking(move || {
            open_oauth_service().request_device_code(
                &flow.device_authorization_endpoint,
                &flow.client_id,
                Some(&scope),
            )
        })
        .await
    };
    let auth = started.map_err(|e| {
        tracing::warn!(target: LOGGER, "Device sign-in for {} did not start: {}", request.catalog_slug, e);
        HTTPException::new(502, format!("Could not start the sign-in: {e}"))
    })?;

    let now = now_ts();
    purge_oauth_state(Some(now));
    let session_id = MCPOAuthService::generate_state();
    PENDING_DEVICE_SESSIONS.lock().insert(
        session_id.clone(),
        PendingDeviceSession {
            project_id: project_id.to_string(),
            catalog_slug: request.catalog_slug.clone(),
            server_url,
            client_id: flow.client_id,
            token_endpoint: flow.token_endpoint,
            device_code: auth.device_code,
            interval: auth.interval,
            next_poll_at: now,
            expires_at: now + auth.expires_in as f64,
        },
    );
    tracing::info!(target: LOGGER, "Device sign-in started for {}", request.catalog_slug);

    Ok(MCPDeviceStartResponse {
        session_id,
        user_code: auth.user_code,
        verification_uri: auth.verification_uri,
        verification_uri_complete: auth.verification_uri_complete,
        interval: auth.interval,
        expires_in: auth.expires_in,
    })
}

/// Poll a device sign-in once.  On a grant the token registers the server
/// exactly as a pasted one would.
///
/// Polls arriving faster than the provider's interval answer "pending"
/// without reaching it, so a hasty client cannot earn a `slow_down`.
pub async fn poll_device_sign_in(
    project_id: &str,
    request: MCPDevicePollRequest,
    model_service: Arc<ModelService>,
) -> Result<MCPDevicePollResponse, HTTPException> {
    let pending = |interval: i64| MCPDevicePollResponse {
        status: "pending".to_string(),
        interval: Some(interval),
        server: None,
    };
    let finished = |status: &str| MCPDevicePollResponse { status: status.to_string(), ..Default::default() };

    let now = now_ts();
    purge_oauth_state(Some(now));
    let session = {
        let mut sessions = PENDING_DEVICE_SESSIONS.lock();
        match sessions.get_mut(&request.session_id).filter(|s| s.project_id == project_id) {
            None => {
                return Err(HTTPException::new(404, "No sign-in is waiting under that session. Start again."));
            }
            Some(session) if now < session.next_poll_at => return Ok(pending(session.interval)),
            Some(session) => {
                // Reserved before the request goes out, so a concurrent poll
                // waits rather than doubling up.
                session.next_poll_at = now + session.interval as f64;
                session.clone()
            }
        }
    };
    let end_session = || PENDING_DEVICE_SESSIONS.lock().remove(&request.session_id);

    let polled = {
        let s = session.clone();
        blocking(move || open_oauth_service().poll_device_token(&s.token_endpoint, &s.client_id, &s.device_code))
            .await
    };
    let token = match polled {
        Ok(DevicePoll::Pending) => return Ok(pending(session.interval)),
        Ok(DevicePoll::SlowDown { interval }) => {
            // RFC 8628 §3.5: without a new interval from the server, add 5s.
            let interval = interval.filter(|i| *i > session.interval).unwrap_or(session.interval + 5);
            if let Some(s) = PENDING_DEVICE_SESSIONS.lock().get_mut(&request.session_id) {
                s.interval = interval;
                s.next_poll_at = now + interval as f64;
            }
            return Ok(pending(interval));
        }
        Ok(DevicePoll::Denied) => {
            end_session();
            return Ok(finished("denied"));
        }
        Ok(DevicePoll::Expired) => {
            end_session();
            return Ok(finished("expired"));
        }
        Err(e) => {
            end_session();
            tracing::warn!(target: LOGGER, "Device sign-in for {} failed: {}", session.catalog_slug, e);
            return Err(HTTPException::new(502, format!("The sign-in failed: {e}")));
        }
        Ok(DevicePoll::Granted(token)) => token,
    };
    end_session();
    tracing::info!(target: LOGGER, "Device sign-in granted for {}", session.catalog_slug);

    let register = MCPRegisterRequest {
        catalog_slug: Some(session.catalog_slug),
        server_url: Some(session.server_url),
        token: Some(token.access_token),
        ..Default::default()
    };
    let registered = register_mcp_server(project_id, register, model_service).await?;
    Ok(MCPDevicePollResponse { status: "connected".to_string(), interval: None, server: registered.server })
}

async fn start_device_sign_in_handler(
    AxumPath(project_id): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: MCPDeviceStartRequest = match parse_body(&headers, &body, DEVICE_START_REQUEST_FIELDS) {
        Ok(request) => request,
        Err(response) => return response,
    };
    match start_device_sign_in(&project_id, request).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => e.into_response(),
    }
}

async fn poll_device_sign_in_handler(
    AxumPath(project_id): AxumPath<String>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let request: MCPDevicePollRequest = match parse_body(&headers, &body, DEVICE_POLL_REQUEST_FIELDS) {
        Ok(request) => request,
        Err(response) => return response,
    };
    match poll_device_sign_in(&project_id, request, get_model_service()).await {
        Ok(response) => Json(response).into_response(),
        Err(e) => e.into_response(),
    }
}

/// `GET`/`DELETE` on a literal segment that shadows `{server_id}`.
///
/// FastAPI tries routes in order and only answers 405 when no route matches
/// both path and method, so `GET .../mcp/register` reaches `get_mcp_server`
/// with `server_id="register"`.  axum stops at the static route, so the
/// literal routes carry these methods and delegate.
fn server_id_fallback(literal: &'static str) -> MethodRouter {
    get(move |AxumPath(project_id): AxumPath<String>| {
        get_mcp_server(AxumPath((project_id, literal.to_string())))
    })
    .delete(move |AxumPath(project_id): AxumPath<String>| {
        delete_mcp_server_from_project(AxumPath((project_id, literal.to_string())))
    })
}

/// The router (`APIRouter(prefix="/v1", tags=["mcp"])`).  Literal routes are
/// registered alongside the `{server_id}` ones; axum prefers static segments,
/// so the literals also carry the `{server_id}` methods (see
/// [`server_id_fallback`]).
pub fn router() -> Router {
    Router::new()
        .route("/v1/projects/{project_id}/mcp", get(list_mcp_servers_for_project))
        .route("/v1/projects/{project_id}/mcp/tools/count", get(count_mcp_tools_for_project))
        .route(
            "/v1/projects/{project_id}/mcp/{server_id}",
            get(get_mcp_server).delete(delete_mcp_server_from_project),
        )
        .route("/v1/projects/{project_id}/mcp/{server_id}/refresh", post(refresh_mcp_server))
        .route(
            "/v1/projects/{project_id}/mcp/register",
            server_id_fallback("register").post(register_mcp_server_handler),
        )
        .route(
            "/v1/projects/{project_id}/mcp/definition",
            server_id_fallback("definition").post(register_mcp_from_definition_handler),
        )
        .route(
            "/v1/projects/{project_id}/mcp/oauth/callback",
            get(mcp_oauth_callback_for_project_handler).post(complete_mcp_oauth_handler),
        )
        .route("/v1/mcp/oauth/callback", get(mcp_oauth_callback_handler))
        .route("/v1/projects/{project_id}/mcp/device/start", post(start_device_sign_in_handler))
        .route("/v1/projects/{project_id}/mcp/device/poll", post(poll_device_sign_in_handler))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlsplit_round_trips_and_info_lands_on_the_path() {
        let mut parts = urlsplit("http://127.0.0.1:61185/v1/mcp?token=abc");
        assert_eq!(parts.scheme, "http");
        assert_eq!(parts.netloc, "127.0.0.1:61185");
        assert_eq!(parts.path, "/v1/mcp");
        assert_eq!(parts.query, "token=abc");
        parts.path = format!("{}/info", parts.path.trim_end_matches('/'));
        assert_eq!(urlunsplit(&parts), "http://127.0.0.1:61185/v1/mcp/info?token=abc");
        let bare = urlsplit("https://mcp.example.com/");
        assert_eq!(urlunsplit(&bare), "https://mcp.example.com/");
    }

    #[test]
    fn urlparse_path_drops_params_of_the_last_segment() {
        let parts = urlsplit("https://h/a;x/b;y?token=t");
        assert_eq!(urlparse_path(&parts), "/a;x/b");
    }

    #[test]
    fn query_tokens_follow_parse_qs() {
        assert_eq!(query_param_first("a=1&token=x%20y", "token").as_deref(), Some("x y"));
        assert_eq!(query_param_first("token=&token=z", "token").as_deref(), Some("z"));
        assert_eq!(query_param_first("token", "token"), None);
    }

    #[test]
    fn escape_and_repr_match_python() {
        assert_eq!(escape("<a href=\"x\">'&'</a>"), "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;");
        assert_eq!(py_repr("abc"), "'abc'");
        assert_eq!(py_repr("it's"), "\"it's\"");
        assert_eq!(py_repr("a'\"b"), "'a\\'\"b'");
        assert_eq!(quote_plus("a b/c"), "a+b%2Fc");
    }

    #[test]
    fn server_response_requires_the_python_keys() {
        let mut m = Map::new();
        m.insert("id".into(), json!("s1"));
        m.insert("name".into(), json!("n"));
        assert_eq!(build_server_response(&m).unwrap_err(), "'enabled'");
        m.insert("enabled".into(), json!(true));
        m.insert("tools".into(), json!([{"name": "t", "extra": 1}]));
        let resp = build_server_response(&m).unwrap();
        assert_eq!(resp.server_type, "local");
        assert_eq!(resp.status, "ready");
        assert_eq!(resp.tools[0].name, "t");
    }

    #[test]
    fn authorization_error_detail_links_only_http_uris() {
        let d = authorization_error_detail("invalid_scope", Some("bad <scope>"), Some("javascript:alert(1)"));
        assert_eq!(d, "<p>invalid_scope</p><p>bad &lt;scope&gt;</p><p>javascript:alert(1)</p>");
        let d = authorization_error_detail("e", None, Some("HTTPS://x.example/doc?a=1&b=2"));
        assert!(d.contains("<a href=\"HTTPS://x.example/doc?a=1&amp;b=2\" rel=\"noopener noreferrer\">"));
    }

    #[test]
    fn success_page_offers_the_return_link() {
        let page = oauth_success_page("Notion", 1, "s 1", Some("smartloop://done?x=1"));
        assert_eq!(page.status_code, 200);
        assert!(page.body.contains("registered with 1 tool."));
        assert!(page.body.contains("href=\"smartloop://done?x=1&amp;server_id=s+1\""));
        let page = oauth_success_page("N", 2, "s", None);
        assert!(page.body.contains("2 tools."));
        assert!(!page.body.contains("Return to Smartloop"));
    }

    // ---------------------------------------------------- routing / validation

    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use serial_test::serial;
    use tower::ServiceExt;

    /// Answers every lookup as Python's `ValueError("MCP server not found: <id>")`
    /// and records which server ids were asked for.
    #[derive(Default)]
    struct FakeRegistry {
        asked: parking_lot::Mutex<Vec<(String, String)>>,
    }

    impl FakeRegistry {
        fn not_found(&self, verb: &str, server_id: &str) -> anyhow::Result<Value> {
            self.asked.lock().push((verb.to_string(), server_id.to_string()));
            Err(ModelServiceError::Value(format!("MCP server not found: {server_id}")).into())
        }
    }

    impl McpServerRegistry for FakeRegistry {
        fn add_local_mcp_server(&self, _: AddLocalMcpServer) -> anyhow::Result<Value> {
            unreachable!("not registered in these tests")
        }
        fn add_remote_mcp_server(&self, _: AddRemoteMcpServer) -> anyhow::Result<Value> {
            unreachable!("not registered in these tests")
        }
        fn list_mcp_servers_with_status(&self, _: &str) -> anyhow::Result<Vec<Value>> {
            Ok(vec![json!({"id": "s1"})])
        }
        fn count_mcp_tools(&self, _: &str) -> anyhow::Result<Value> {
            Ok(json!({"tools": 3, "by_server": {}}))
        }
        fn mcp_server_status(&self, server_id: &str, _: &str) -> anyhow::Result<Value> {
            self.not_found("get", server_id)
        }
        fn retry_mcp_server(&self, server_id: &str, _: &str) -> anyhow::Result<Value> {
            self.not_found("refresh", server_id)
        }
        fn delete_mcp_server(&self, server_id: &str, _: &str) -> anyhow::Result<Value> {
            self.not_found("delete", server_id)
        }
    }

    /// Installs the fake registry for the test's duration.
    struct RegistryGuard;

    impl Drop for RegistryGuard {
        fn drop(&mut self) {
            set_hooks(McpRouteHooks::default());
        }
    }

    fn install_fake() -> (Arc<FakeRegistry>, RegistryGuard) {
        let fake = Arc::new(FakeRegistry::default());
        set_hooks(McpRouteHooks { mcp_registry: Some(fake.clone()), ..Default::default() });
        (fake, RegistryGuard)
    }

    async fn call(method: &str, uri: &str, content_type: Option<&str>, body: &str) -> (u16, Value) {
        let mut req = Request::builder().method(method).uri(uri);
        if let Some(ct) = content_type {
            req = req.header("content-type", ct);
        }
        let resp = router().oneshot(req.body(Body::from(body.to_string())).unwrap()).await.unwrap();
        let status = resp.status().as_u16();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
    }

    #[tokio::test]
    #[serial]
    async fn literal_routes_win_for_their_own_method() {
        let (_fake, _guard) = install_fake();
        let (status, body) = call("GET", "/v1/projects/p1/mcp/tools/count", None, "").await;
        assert_eq!(status, 200);
        assert_eq!(body["tools"], json!(3));
        let (status, body) = call("GET", "/v1/projects/p1/mcp", None, "").await;
        assert_eq!(status, 200);
        assert_eq!(body, json!({"servers": [{"id": "s1"}]}));
    }

    #[tokio::test]
    #[serial]
    async fn other_methods_on_literals_reach_the_server_id_routes() {
        // FastAPI matches `GET .../mcp/register` to `get_mcp_server` with
        // server_id="register" rather than answering 405.
        let (fake, _guard) = install_fake();
        let (status, body) = call("GET", "/v1/projects/p1/mcp/register", None, "").await;
        assert_eq!(status, 404);
        assert_eq!(body, json!({"detail": "MCP server not found: register"}));
        let (status, _) = call("DELETE", "/v1/projects/p1/mcp/definition", None, "").await;
        assert_eq!(status, 404);
        let (status, _) = call("GET", "/v1/projects/p1/mcp/tools", None, "").await;
        assert_eq!(status, 404);
        let (status, _) = call("POST", "/v1/projects/p1/mcp/tools/refresh", None, "").await;
        assert_eq!(status, 404);
        let (status, _) = call("POST", "/v1/projects/p1/mcp/oauth/refresh", None, "").await;
        assert_eq!(status, 404);
        assert_eq!(
            *fake.asked.lock(),
            vec![
                ("get".to_string(), "register".to_string()),
                ("delete".to_string(), "definition".to_string()),
                ("get".to_string(), "tools".to_string()),
                ("refresh".to_string(), "tools".to_string()),
                ("refresh".to_string(), "oauth".to_string()),
            ]
        );
    }

    #[tokio::test]
    #[serial]
    async fn bodies_that_do_not_validate_get_fastapis_422() {
        let (_fake, _guard) = install_fake();
        let uri = "/v1/projects/p1/mcp/definition";
        let (status, body) = call("POST", uri, Some("application/json"), "{}").await;
        assert_eq!(status, 422);
        assert_eq!(
            body,
            json!({"detail": [{"type": "missing", "loc": ["body", "toml"], "msg": "Field required", "input": {}}]})
        );

        let (status, body) = call("POST", uri, Some("application/json"), "{\"toml\": 5}").await;
        assert_eq!(status, 422);
        assert_eq!(body["detail"][0]["type"], json!("string_type"));
        assert_eq!(body["detail"][0]["loc"], json!(["body", "toml"]));

        let (status, body) = call("POST", uri, Some("application/json"), "{\"toml\": ").await;
        assert_eq!(status, 422);
        assert_eq!(body["detail"][0]["type"], json!("json_invalid"));
        assert_eq!(body["detail"][0]["msg"], json!("JSON decode error"));

        let (status, body) = call("POST", uri, Some("application/json"), "[1]").await;
        assert_eq!(status, 422);
        assert_eq!(body["detail"][0]["type"], json!("model_attributes_type"));

        let (status, body) = call("POST", uri, Some("text/plain"), "{\"toml\": \"x\"}").await;
        assert_eq!(status, 422);
        assert_eq!(body["detail"][0]["type"], json!("model_attributes_type"));

        let (status, body) = call("POST", uri, Some("application/json"), "").await;
        assert_eq!(status, 422);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
        assert_eq!(body["detail"][0]["type"], json!("missing"));

        let register = "/v1/projects/p1/mcp/register";
        let (status, body) =
            call("POST", register, Some("application/json"), "{\"args\": [\"ok\", 1], \"env\": []}").await;
        assert_eq!(status, 422);
        assert_eq!(body["detail"][0]["loc"], json!(["body", "args", 1]));
        assert_eq!(body["detail"][1]["type"], json!("dict_type"));

        let oauth = "/v1/projects/p1/mcp/oauth/callback";
        let (status, body) = call("POST", oauth, Some("application/json"), "{\"code\": \"c\"}").await;
        assert_eq!(status, 422);
        let missing: Vec<Value> = body["detail"].as_array().unwrap().iter().map(|e| e["loc"][1].clone()).collect();
        assert_eq!(
            missing,
            vec![
                json!("server_url"),
                json!("state"),
                json!("code_verifier"),
                json!("client_id"),
                json!("token_endpoint"),
                json!("authorization_endpoint"),
            ]
        );
    }

    #[test]
    #[serial]
    fn purge_stamps_unstamped_sessions_and_evicts_expired_ones() {
        PENDING_OAUTH_SESSIONS.lock().clear();
        COMPLETED_OAUTH_RESPONSES.lock().clear();
        let mut fresh = Map::new();
        fresh.insert("project_id".into(), json!("p"));
        PENDING_OAUTH_SESSIONS.lock().insert("fresh".into(), fresh);
        let mut old = Map::new();
        old.insert("created_at".into(), json!(1000.0));
        PENDING_OAUTH_SESSIONS.lock().insert("old".into(), old);
        for i in 0..(OAUTH_RESPONSE_MAX + 3) {
            COMPLETED_OAUTH_RESPONSES.lock().insert(format!("s{i}"), (5000.0, HtmlPage::ok("x")));
        }
        COMPLETED_OAUTH_RESPONSES.lock().insert("stale".into(), (1500.0, HtmlPage::ok("x")));

        purge_oauth_state(Some(1000.0 + OAUTH_SESSION_TTL_S + 1.0));
        {
            let sessions = PENDING_OAUTH_SESSIONS.lock();
            assert!(!sessions.contains_key("old"));
            assert_eq!(sessions["fresh"]["created_at"], json!(1000.0 + OAUTH_SESSION_TTL_S + 1.0));
        }
        {
            let responses = COMPLETED_OAUTH_RESPONSES.lock();
            assert!(!responses.contains_key("stale"));
            assert_eq!(responses.len(), OAUTH_RESPONSE_MAX);
            // Oldest first out.
            assert!(!responses.contains_key("s0") && responses.contains_key("s3"));
        }
        PENDING_OAUTH_SESSIONS.lock().clear();
        COMPLETED_OAUTH_RESPONSES.lock().clear();
    }

    #[test]
    #[serial]
    fn issuer_mismatch_and_omission_are_refused() {
        PENDING_OAUTH_SESSIONS.lock().clear();
        let mut session = Map::new();
        session.insert("issuer".into(), json!("https://honest.example"));
        session.insert("issuer_parameter_supported".into(), json!(true));
        PENDING_OAUTH_SESSIONS.lock().insert("st".into(), session);

        assert!(reject_unexpected_issuer(Some("st"), Some("https://honest.example")).is_none());
        let page = reject_unexpected_issuer(Some("st"), Some("https://evil.example")).unwrap();
        assert_eq!(page.status_code, 400);
        assert!(page.body.contains(
            "expected &#x27;https://honest.example&#x27;, got &#x27;https://evil.example&#x27;"
        ));
        let page = reject_unexpected_issuer(Some("st"), None).unwrap();
        assert!(page.body.contains("the response carried none"));
        // Unknown state: nothing to compare, handled as an expired session.
        assert!(reject_unexpected_issuer(Some("other"), Some("x")).is_none());
        PENDING_OAUTH_SESSIONS.lock().clear();
    }

    #[test]
    #[serial]
    fn scope_retry_happens_exactly_once() {
        PENDING_OAUTH_SESSIONS.lock().clear();
        let mut session = Map::new();
        session.insert("scope".into(), json!("a b"));
        session.insert("code_challenge".into(), json!("cc"));
        session.insert("authorization_endpoint".into(), json!("https://as.example/authorize"));
        session.insert("client_id".into(), json!("cid"));
        session.insert("callback_url".into(), json!("http://localhost:1/cb"));
        PENDING_OAUTH_SESSIONS.lock().insert("st".into(), session);

        assert!(retry_authorization_without_scope(Some("st"), "access_denied").is_none());
        let url = retry_authorization_without_scope(Some("st"), "invalid_scope").unwrap();
        assert!(url.starts_with("https://as.example/authorize"));
        assert!(!url.contains("scope="));
        {
            let sessions = PENDING_OAUTH_SESSIONS.lock();
            assert_eq!(sessions["st"]["scope_retry"], json!(true));
            assert_eq!(sessions["st"]["scope"], Value::Null);
        }
        assert!(retry_authorization_without_scope(Some("st"), "invalid_scope").is_none());
        PENDING_OAUTH_SESSIONS.lock().clear();
    }
}
