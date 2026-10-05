//! Port of `smartloop/platform/client.py`.
//!
//! HTTP client for the Smartloop platform API (api.smartloop.ai).
//!
//! Handles only JSON control-plane calls — no file downloading.  Calls are
//! synchronous (`reqwest::blocking`), mirroring the Python `httpx` client;
//! async callers run them on the blocking pool.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

#[allow(unused_imports)]
use crate::platform::auth as platform_auth;

/// Python logger name for this module.
pub const LOGGER: &str = "smartloop.platform.client";

// URL path constants
pub const BOOTSTRAP_MODELS: &str = "/v1/bootstrap/models";
/// `{}` is replaced with the model name.
pub const MODEL_DOWNLOAD: &str = "/v1/models/{}/download";
pub const USAGE_TOKENS: &str = "/v1/usage/tokens";
pub const USERS_ME: &str = "/v1/users/me";
pub const JWT_REFRESH: &str = "/v1/auth/jwt/refresh";
/// `{}` is replaced with the model name.
pub const MODEL_UPLOAD: &str = "/v1/models/{}/upload";
pub const MODELS: &str = "/v1/models";
pub const RELEASES_LATEST: &str = "/v1/releases/latest";
pub const MCP_CATALOG: &str = "/v1/mcp-catalog";
pub const WEBSEARCH: &str = "/v1/websearch";
pub const WEBSEARCH_STATUS: &str = "/v1/websearch/status";

/// Control-plane request timeout (seconds).
pub const TIMEOUT: u64 = 30;

/// Web search sits in front of a user-visible turn, so it gets a tighter
/// budget than the control-plane calls: a slow search is worth abandoning,
/// and the status check runs ahead of every query.
pub const SEARCH_TIMEOUT: u64 = 15;
pub const SEARCH_STATUS_TIMEOUT: u64 = 5;

/// The refresh grant is a single small POST, and it lands *inside* another
/// call's budget — the web-search gate allows itself 5s precisely because it
/// sits in front of a user-visible turn, and inheriting the 30s control-plane
/// timeout let one 401 turn that into 40s.
pub const REFRESH_TIMEOUT: u64 = 10;

/// HTTP header map (same type `credential_store::auth_headers` returns).
pub type Headers = BTreeMap<String, String>;
/// A JSON object (Python `dict`).
pub type JsonObject = serde_json::Map<String, Value>;
/// Returns the auth headers to send.
pub type AuthHeadersFn = Arc<dyn Fn() -> Headers + Send + Sync>;
/// Redeems the refresh token; returns whether a new access token was stored.
pub type RefreshFn = Arc<dyn Fn() -> bool + Send + Sync>;

/// Identify this client to the API: `smartloop/<version> (<os>)`.
///
/// The platform records each call in the account's access log; naming
/// ourselves lets the owner tell their own agent apart from anything else
/// using their token.  Deliberately not "smartloop-cli": these calls come from
/// the local API server, background agents and the downloader as much as from
/// the CLI.  The OS part matches Python's `platform.system().lower()`
/// (`darwin`, `linux`, `windows`).
pub fn user_agent() -> String {
    todo!()
}

/// The account's plan does not grant this model (HTTP 403).
///
/// A kind of permission error (see [`PlatformError::is_permission`]) so existing
/// handlers keep catching it, but distinguishable from a 401: a 401 means
/// "sign in", a 403 means the caller reached the endpoint and the model is not
/// on their plan.  Callers map it to an HTTP 403 so the UI can offer an upgrade
/// instead of a login.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct ModelAccessError {
    pub message: String,
    /// Always 403.
    pub status_code: u16,
}

impl ModelAccessError {
    /// A new 403 model-access error with *message*.
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), status_code: 403 }
    }
}

/// Which Python exception class a refusal maps to for an endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorClass {
    /// Python `ConnectionError` — transient, retryable.
    Connection,
    /// Python `PermissionError` — a verdict about this caller.
    Permission,
}

/// Errors raised by platform calls, mirroring the Python exception classes.
///
/// `status_code` carries the upstream HTTP status when there was one, so
/// callers can classify a failure without parsing the message.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum PlatformError {
    /// Python `ConnectionError`: transport failure, 429 or 5xx.
    #[error("{message}")]
    Connection { message: String, status_code: Option<u16> },
    /// Python `PermissionError`: 401/403/404-style refusals.
    #[error("{message}")]
    Permission { message: String, status_code: Option<u16> },
    /// Python `ModelAccessError` (a `PermissionError` subclass).
    #[error(transparent)]
    ModelAccess(#[from] ModelAccessError),
    /// Python `ValueError`: an unusable successful response.
    #[error("{0}")]
    Value(String),
    /// Any other unexpected failure (bad JSON, local DB error, ...).
    #[error("{0}")]
    Other(String),
}

impl PlatformError {
    /// A `ConnectionError` with no status.
    pub fn connection(message: impl Into<String>) -> Self {
        Self::Connection { message: message.into(), status_code: None }
    }

    /// A `PermissionError` with no status.
    pub fn permission(message: impl Into<String>) -> Self {
        Self::Permission { message: message.into(), status_code: None }
    }

    /// True for `ConnectionError`.
    pub fn is_connection(&self) -> bool {
        matches!(self, Self::Connection { .. })
    }

    /// True for `PermissionError`, including [`ModelAccessError`].
    pub fn is_permission(&self) -> bool {
        matches!(self, Self::Permission { .. } | Self::ModelAccess(_))
    }

    /// The upstream HTTP status, when the error came from a response.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::Connection { status_code, .. } | Self::Permission { status_code, .. } => *status_code,
            Self::ModelAccess(e) => Some(e.status_code),
            Self::Value(_) | Self::Other(_) => None,
        }
    }

    /// The human-readable message (`str(exc)` in Python).
    pub fn message(&self) -> String {
        self.to_string()
    }
}

/// Result alias for platform calls.
pub type PlatformResult<T> = Result<T, PlatformError>;

/// Whether *status_code* says "try again" rather than "this caller may not".
///
/// A 429 is a rate limit and a 5xx is the platform having a bad moment.
/// Neither is a verdict about the credentials, so neither may be reported as one.
pub fn is_transient(status_code: u16) -> bool {
    status_code == 429 || status_code >= 500
}

/// Raise an *error_cls* error for a non-2xx response, tagged with the status.
///
/// *error_cls* names what a refusal means for this endpoint, but a 429 or 5xx
/// is always a `ConnectionError`.  Callers act on the distinction — the refresh
/// agent clears both tokens on a `PermissionError`, and the cache-through
/// service only serves stale data for a `ConnectionError`.  The message comes
/// from *error_map* when it has the status, else
/// `"API error ({status}): {reason_phrase}"`.
pub fn raise_for_status(
    status: u16,
    error_map: &[(u16, String)],
    error_cls: ErrorClass,
) -> PlatformResult<()> {
    let _ = (status, error_map, error_cls);
    todo!()
}

/// The platform operations the rest of the app (and tests) depend on.
///
/// Implemented by [`SLPPlatformClient`]; tests substitute fakes.  Every method
/// has the semantics documented on the matching `SLPPlatformClient` method.
/// Default bodies report the operation as unsupported so a fake only needs to
/// implement what it exercises.
pub trait PlatformApi: Send + Sync {
    fn list_bootstrap_models(&self) -> PlatformResult<Vec<JsonObject>> {
        Err(unsupported("list_bootstrap_models"))
    }
    fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
        Err(unsupported("list_models"))
    }
    fn get_latest_release(&self, channel: &str) -> PlatformResult<Option<JsonObject>> {
        let _ = channel;
        Err(unsupported("get_latest_release"))
    }
    fn list_mcp_catalog(&self) -> PlatformResult<Vec<JsonObject>> {
        Err(unsupported("list_mcp_catalog"))
    }
    fn get_model_artifacts(&self, model_name: &str) -> PlatformResult<Vec<(String, String)>> {
        let _ = model_name;
        Err(unsupported("get_model_artifacts"))
    }
    fn get_current_user(&self) -> PlatformResult<JsonObject> {
        Err(unsupported("get_current_user"))
    }
    fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        let _ = payload;
        Err(unsupported("update_current_user"))
    }
    fn delete_current_user(&self) -> PlatformResult<()> {
        Err(unsupported("delete_current_user"))
    }
    fn refresh_access_token(&self, refresh_token: &str) -> PlatformResult<JsonObject> {
        let _ = refresh_token;
        Err(unsupported("refresh_access_token"))
    }
    fn get_model_upload_url(&self, model_name: &str, filename: &str) -> PlatformResult<JsonObject> {
        let _ = (model_name, filename);
        Err(unsupported("get_model_upload_url"))
    }
    fn push_token_usage(&self, record: &JsonObject, timeout: u64) {
        let _ = (record, timeout);
    }
    fn web_search(&self, query: &str, count: u32, timeout: u64) -> PlatformResult<Vec<String>> {
        let _ = (query, count, timeout);
        Err(unsupported("web_search"))
    }
    fn web_search_status(&self, timeout: u64) -> PlatformResult<bool> {
        let _ = timeout;
        Err(unsupported("web_search_status"))
    }
}

fn unsupported(op: &str) -> PlatformError {
    PlatformError::Other(format!("{op} is not supported by this platform client"))
}

/// Thin HTTP client for Smartloop platform API calls.
///
/// * `base_url` — API base URL, e.g. `https://api.smartloop.ai` (trailing `/`
///   stripped).
/// * `auth_headers_fn` — returns auth headers; defaults to
///   [`platform_auth::auth_headers`].
/// * `refresh_fn` — redeems the refresh token once per 401 in [`Self::authed`];
///   defaults to [`Self::refresh_credentials`].  May legitimately decline.
#[derive(Clone)]
pub struct SLPPlatformClient {
    base_url: String,
    auth_headers_fn: AuthHeadersFn,
    refresh_fn: Option<RefreshFn>,
}

impl std::fmt::Debug for SLPPlatformClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SLPPlatformClient").field("base_url", &self.base_url).finish()
    }
}

impl SLPPlatformClient {
    /// Client with the default auth headers and refresh hook.
    pub fn new(base_url: &str) -> Self {
        let _ = base_url;
        todo!()
    }

    /// Client with explicit hooks (`None` selects the default for each).
    pub fn with_hooks(
        base_url: &str,
        auth_headers_fn: Option<AuthHeadersFn>,
        refresh_fn: Option<RefreshFn>,
    ) -> Self {
        let _ = (base_url, auth_headers_fn, refresh_fn);
        todo!()
    }

    /// The normalized base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Redeem the stored refresh token for a new access token, against *this*
    /// client (via `token_agent::refresh_access_token_now`).
    pub fn refresh_credentials(&self) -> bool {
        todo!()
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Request headers: always `User-Agent`, with *auth* merged in.
    pub fn headers(&self, auth: Option<&Headers>) -> Headers {
        let _ = auth;
        todo!()
    }

    /// Send an authenticated request, redeeming the refresh token once on a 401.
    ///
    /// *send* is called with the headers to use.  After a 401 the refresh hook
    /// runs, auth headers are re-read, and the request is retried only if the
    /// credentials actually changed; otherwise the original 401 is returned.
    pub fn authed<F>(&self, send: F) -> reqwest::Result<reqwest::blocking::Response>
    where
        F: Fn(&Headers) -> reqwest::Result<reqwest::blocking::Response>,
    {
        let _ = send;
        todo!()
    }

    fn http(&self, timeout: u64) -> reqwest::blocking::Client {
        let _ = (timeout, Duration::from_secs(0), self.url(""));
        todo!()
    }

    /// GET /v1/bootstrap/models (no auth). Returns the array of model entries.
    pub fn list_bootstrap_models(&self) -> PlatformResult<Vec<JsonObject>> {
        let _ = self.http(TIMEOUT);
        todo!()
    }

    /// GET /v1/models (auth optional). Falls back to an anonymous request when
    /// the credentials are still refused after the refresh attempt.
    pub fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
        todo!()
    }

    /// GET /v1/releases/latest?channel=... (no auth). Returns the
    /// `LatestReleaseResponse` wrapper `{update_available, update_required, release}`.
    pub fn get_latest_release(&self, channel: &str) -> PlatformResult<Option<JsonObject>> {
        let _ = channel;
        todo!()
    }

    /// GET /v1/mcp-catalog (auth required). Curated MCP server catalog entries.
    pub fn list_mcp_catalog(&self) -> PlatformResult<Vec<JsonObject>> {
        todo!()
    }

    /// GET /v1/models/{model_name}/download (auth optional).
    ///
    /// Returns the `{filename: signed_url}` map in server order.  403 →
    /// [`ModelAccessError`]; 401/404 → `PermissionError`; 429/5xx →
    /// `ConnectionError`; empty artifacts → `ValueError`.
    pub fn get_model_artifacts(&self, model_name: &str) -> PlatformResult<Vec<(String, String)>> {
        let _ = model_name;
        todo!()
    }

    /// GET /v1/users/me (auth required). Returns user profile dict.
    pub fn get_current_user(&self) -> PlatformResult<JsonObject> {
        todo!()
    }

    /// PATCH /v1/users/me (auth required). Returns updated user profile dict.
    pub fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        let _ = payload;
        todo!()
    }

    /// DELETE /v1/users/me (auth required). Permanently deletes the account.
    pub fn delete_current_user(&self) -> PlatformResult<()> {
        todo!()
    }

    /// POST /v1/auth/jwt/refresh (no auth). Returns `access_token` and
    /// optionally `refresh_token`; missing `access_token` → `ValueError`.
    pub fn refresh_access_token(&self, refresh_token: &str) -> PlatformResult<JsonObject> {
        let _ = refresh_token;
        todo!()
    }

    /// POST /v1/models/{name}/upload (auth required). Returns `{url, blob_path, expires_at}`.
    pub fn get_model_upload_url(&self, model_name: &str, filename: &str) -> PlatformResult<JsonObject> {
        let _ = (model_name, filename);
        todo!()
    }

    /// POST /v1/usage/tokens (auth required). Fire-and-forget telemetry;
    /// failures are logged, never raised.  Deliberately not through
    /// [`Self::authed`] so a telemetry 401 can never log the user out.
    pub fn push_token_usage(&self, record: &JsonObject, timeout: u64) {
        let _ = (record, timeout);
        todo!()
    }

    /// GET /v1/websearch?q=..&count=.. (auth required). Returns result URLs.
    /// 401/403/429 → `PermissionError`; other failures → `ConnectionError`.
    pub fn web_search(&self, query: &str, count: u32, timeout: u64) -> PlatformResult<Vec<String>> {
        let _ = (query, count, timeout);
        todo!()
    }

    /// GET /v1/websearch/status (auth required). Whether search would serve.
    /// 401/403 → `PermissionError`; anything else → `ConnectionError`.
    pub fn web_search_status(&self, timeout: u64) -> PlatformResult<bool> {
        let _ = timeout;
        todo!()
    }
}

impl PlatformApi for SLPPlatformClient {
    fn list_bootstrap_models(&self) -> PlatformResult<Vec<JsonObject>> {
        SLPPlatformClient::list_bootstrap_models(self)
    }
    fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
        SLPPlatformClient::list_models(self)
    }
    fn get_latest_release(&self, channel: &str) -> PlatformResult<Option<JsonObject>> {
        SLPPlatformClient::get_latest_release(self, channel)
    }
    fn list_mcp_catalog(&self) -> PlatformResult<Vec<JsonObject>> {
        SLPPlatformClient::list_mcp_catalog(self)
    }
    fn get_model_artifacts(&self, model_name: &str) -> PlatformResult<Vec<(String, String)>> {
        SLPPlatformClient::get_model_artifacts(self, model_name)
    }
    fn get_current_user(&self) -> PlatformResult<JsonObject> {
        SLPPlatformClient::get_current_user(self)
    }
    fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        SLPPlatformClient::update_current_user(self, payload)
    }
    fn delete_current_user(&self) -> PlatformResult<()> {
        SLPPlatformClient::delete_current_user(self)
    }
    fn refresh_access_token(&self, refresh_token: &str) -> PlatformResult<JsonObject> {
        SLPPlatformClient::refresh_access_token(self, refresh_token)
    }
    fn get_model_upload_url(&self, model_name: &str, filename: &str) -> PlatformResult<JsonObject> {
        SLPPlatformClient::get_model_upload_url(self, model_name, filename)
    }
    fn push_token_usage(&self, record: &JsonObject, timeout: u64) {
        SLPPlatformClient::push_token_usage(self, record, timeout)
    }
    fn web_search(&self, query: &str, count: u32, timeout: u64) -> PlatformResult<Vec<String>> {
        SLPPlatformClient::web_search(self, query, count, timeout)
    }
    fn web_search_status(&self, timeout: u64) -> PlatformResult<bool> {
        SLPPlatformClient::web_search_status(self, timeout)
    }
}
