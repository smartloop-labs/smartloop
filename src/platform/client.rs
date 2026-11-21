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

use reqwest::Method;
use serde_json::Value;

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
pub const DEVICE_CODE: &str = "/v1/auth/device/code";
pub const DEVICE_TOKEN: &str = "/v1/auth/device/token";
/// RFC 8628's grant type for polling with a device code.
pub const DEVICE_GRANT_TYPE: &str = "urn:ietf:params:oauth:grant-type:device_code";
/// `{}` is replaced with the model name.
pub const MODEL_UPLOAD: &str = "/v1/models/{}/upload";
pub const MODELS: &str = "/v1/models";
pub const RELEASES_LATEST: &str = "/v1/releases/latest";
pub const MCP_CATALOG: &str = "/v1/mcp-catalog";
pub const WEBSEARCH: &str = "/v1/websearch";
pub const WEBSEARCH_STATUS: &str = "/v1/websearch/status";
pub const CHAT_QUOTA: &str = "/v1/chat/quota";
pub const APPLE_VERIFY: &str = "/v1/payments/apple/verify";
pub const CHECKOUT_SESSION: &str = "/v1/payments/create-checkout-session";

/// Control-plane request timeout (seconds).
pub const TIMEOUT: u64 = 30;

/// Web search sits in front of a user-visible turn, so it gets a tighter
/// budget than the control-plane calls: a slow search is worth abandoning,
/// and the status check runs ahead of every query.
pub const SEARCH_TIMEOUT: u64 = 15;
pub const SEARCH_STATUS_TIMEOUT: u64 = 5;
/// The cloud allowance is read for a Settings page, not a turn.
pub const CLOUD_USAGE_TIMEOUT: u64 = 10;

/// The refresh grant is a single small POST, and it lands *inside* another
/// call's budget — the web-search gate allows itself 5s precisely because it
/// sits in front of a user-visible turn, and inheriting the 30s control-plane
/// timeout let one 401 turn that into 40s.
pub const REFRESH_TIMEOUT: u64 = 10;

/// Default timeout of [`SLPPlatformClient::push_token_usage`] (seconds).
pub const PUSH_TOKEN_USAGE_TIMEOUT: u64 = 10;

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
/// The platform records each call in the account's access log, and the HTTP
/// library's default agent would make every entry look like an anonymous
/// script.  Naming ourselves lets the owner tell their own agent apart from
/// anything else using their token.
///
/// Deliberately not "smartloop-cli": these calls come from the local API
/// server, background agents and the downloader as much as from the CLI.
/// What the log needs from this header is the product and version — *which*
/// caller it was is already the path on the same log line.  The OS part
/// matches Python's `platform.system().lower()` (`darwin`, `linux`,
/// `windows`).
pub fn user_agent() -> String {
    let os = match std::env::consts::OS {
        "macos" => "darwin",
        other => other,
    };
    format!("smartloop/{} ({})", crate::VERSION, os)
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

/// One poll of a device code sign-in (RFC 8628 §3.5): either the token pair,
/// or where the sign-in stands.  Failures that end it for another reason
/// (an unknown code) are errors.
#[derive(Debug, Clone, PartialEq)]
pub enum DevicePoll {
    /// Approved: the token pair, as every other sign-in returns it.
    Tokens(JsonObject),
    /// Not approved yet; poll again after the interval.
    Pending,
    /// Polled too soon; add 5 seconds to the interval.
    SlowDown,
    /// The user declined.
    Denied,
    /// Nobody approved it in time.
    Expired,
}

/// Whether *status_code* says "try again" rather than "this caller may not".
///
/// A 429 is a rate limit and a 5xx is the platform having a bad moment.
/// Neither is a verdict about the credentials, so neither may be reported as one.
pub fn is_transient(status_code: u16) -> bool {
    status_code == 429 || status_code >= 500
}

/// The standard reason phrase for *status* (httpx's `reason_phrase`), or `""`
/// for a status without one.
fn reason_phrase(status: u16) -> &'static str {
    reqwest::StatusCode::from_u16(status)
        .ok()
        .and_then(|s| s.canonical_reason())
        .unwrap_or("")
}

/// Raise an *error_cls* error for a non-2xx response, tagged with the status.
///
/// The status is attached as `status_code` on the error so callers can
/// classify the failure without re-deriving it from the message.  The API
/// router turns it back into an HTTP status for its own client, and a message
/// is a bad channel for that: it is written for a person to read, and it gets
/// reworded.
///
/// *error_cls* names what a refusal means for this endpoint, but it does not
/// get to claim a transient failure as one: a 429 or 5xx is always a
/// `ConnectionError`.  Callers act on the distinction — the refresh agent
/// clears both tokens on a `PermissionError`, and the cache-through service
/// only serves stale data for a `ConnectionError` — so letting a platform blip
/// arrive as "your credentials are bad" logs the user out over an outage.  The
/// message comes from *error_map* when it has the status, else
/// `"API error ({status}): {reason_phrase}"`.
pub fn raise_for_status(
    status: u16,
    error_map: &[(u16, String)],
    error_cls: ErrorClass,
) -> PlatformResult<()> {
    if (200..300).contains(&status) {
        return Ok(());
    }
    let error_cls = if is_transient(status) { ErrorClass::Connection } else { error_cls };
    let message = match error_map.iter().find(|(code, _)| *code == status) {
        Some((_, message)) => message.clone(),
        None => format!("API error ({}): {}", status, reason_phrase(status)),
    };
    Err(match error_cls {
        ErrorClass::Connection => PlatformError::Connection { message, status_code: Some(status) },
        ErrorClass::Permission => PlatformError::Permission { message, status_code: Some(status) },
    })
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
    fn web_search_status_detail(&self, timeout: u64) -> PlatformResult<JsonObject> {
        let _ = timeout;
        Err(unsupported("web_search_status_detail"))
    }
    fn cloud_quota(&self, timeout: u64) -> PlatformResult<JsonObject> {
        let _ = timeout;
        Err(unsupported("cloud_quota"))
    }
    fn verify_apple_transaction(&self, transaction_id: &str) -> PlatformResult<JsonObject> {
        let _ = transaction_id;
        Err(unsupported("verify_apple_transaction"))
    }
    fn create_checkout_session(&self) -> PlatformResult<JsonObject> {
        Err(unsupported("create_checkout_session"))
    }
    fn request_device_code(&self, client_id: &str) -> PlatformResult<JsonObject> {
        let _ = client_id;
        Err(unsupported("request_device_code"))
    }
    fn poll_device_token(&self, device_code: &str, client_id: &str) -> PlatformResult<DevicePoll> {
        let _ = (device_code, client_id);
        Err(unsupported("poll_device_token"))
    }
}

fn unsupported(op: &str) -> PlatformError {
    PlatformError::Other(format!("{op} is not supported by this platform client"))
}

/// A response whose body has been read in full.
///
/// httpx reads the body inside `httpx.get(...)`, so a failure while reading it
/// is a transport error like any other; reading it here keeps it inside the
/// same `ConnectionError` mapping.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// The HTTP status code.
    pub status: u16,
    /// The raw response body.
    pub body: bytes::Bytes,
}

impl HttpResponse {
    /// Decode the body as JSON (`resp.json()`).  A body that is not JSON is a
    /// `ValueError` in Python (`json.JSONDecodeError`), and so is it here.
    pub fn json(&self) -> PlatformResult<Value> {
        serde_json::from_slice(&self.body)
            .map_err(|e| PlatformError::Value(format!("Invalid JSON in response: {e}")))
    }
}

/// Python truthiness of a JSON value (`bool(value)`).
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `ConnectionError(f"Connection error: {e}")` for a transport failure.
fn connection_error(e: &reqwest::Error) -> PlatformError {
    PlatformError::connection(format!("Connection error: {e}"))
}

/// A JSON array of objects (a `list[dict]` response).
fn object_list(value: Value) -> PlatformResult<Vec<JsonObject>> {
    match value {
        Value::Array(items) => items
            .into_iter()
            .map(|item| match item {
                Value::Object(map) => Ok(map),
                other => Err(PlatformError::Value(format!(
                    "Expected a JSON object in the response list, got {other}"
                ))),
            })
            .collect(),
        other => Err(PlatformError::Value(format!("Expected a JSON list response, got {other}"))),
    }
}

/// A JSON object (a `dict` response).
fn object(value: Value) -> PlatformResult<JsonObject> {
    match value {
        Value::Object(map) => Ok(map),
        other => Err(PlatformError::Value(format!("Expected a JSON object response, got {other}"))),
    }
}

/// The `detail` string of a FastAPI error body, when it has one.
///
/// A refusal whose reason is written for the user (an Apple transaction that
/// is already linked to another account, say) has to reach them as written,
/// so the endpoints that relay one read it back rather than substituting a
/// message of their own.  A non-string `detail` (a validation error list) or
/// a body that is not JSON yields `None`.
pub fn error_detail(body: &[u8]) -> Option<String> {
    match serde_json::from_slice::<Value>(body).ok()?.get("detail")? {
        Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        _ => None,
    }
}

/// The `error` and `error_description` of an OAuth error body (RFC 6749
/// §5.2), which the device sign-in endpoints answer with instead of `detail`.
fn oauth_error(body: &[u8]) -> Option<(String, String)> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    let error = value.get("error")?.as_str()?.to_string();
    let description = value
        .get("error_description")
        .and_then(Value::as_str)
        .filter(|d| !d.trim().is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| error.clone());
    Some((error, description))
}

/// Thin HTTP client for Smartloop platform API calls.
///
/// * `base_url` — API base URL, e.g. `https://api.smartloop.ai` (trailing `/`
///   stripped).
/// * `auth_headers_fn` — returns auth headers; defaults to
///   [`platform_auth::auth_headers`].
/// * `refresh_fn` — redeems the refresh token, returning whether a new access
///   token was stored.  Called once per 401 by [`Self::authed`]; defaults to
///   [`Self::refresh_credentials`].  May legitimately decline and do nothing —
///   a per-project agent child does, because redeeming a single-use token
///   belongs to the process that owns the credential store.  [`Self::authed`]
///   handles that without needing to know.
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
        Self::with_hooks(base_url, None, None)
    }

    /// Client with explicit hooks (`None` selects the default for each).
    pub fn with_hooks(
        base_url: &str,
        auth_headers_fn: Option<AuthHeadersFn>,
        refresh_fn: Option<RefreshFn>,
    ) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_headers_fn: auth_headers_fn.unwrap_or_else(|| Arc::new(platform_auth::auth_headers)),
            refresh_fn,
        }
    }

    /// The normalized base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Redeem the stored refresh token for a new access token.
    ///
    /// Redeems against *this* client, so the refresh goes to the host that
    /// just answered 401 rather than to whatever `AppSettings` names — the two
    /// are the same in production, and only one of them is true in a test.
    ///
    /// No recursion: [`Self::refresh_access_token`] is unauthenticated and
    /// does not route through [`Self::authed`].
    pub fn refresh_credentials(&self) -> bool {
        crate::platform::token_agent::refresh_access_token_now(Some(self))
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Request headers.  Every request identifies itself; *auth* is merged in.
    ///
    /// The endpoints that take no auth pass nothing rather than flipping a
    /// flag — they are a minority, and which of them is unauthenticated is
    /// worth seeing at the call site.
    pub fn headers(&self, auth: Option<&Headers>) -> Headers {
        let mut headers = Headers::new();
        headers.insert("User-Agent".to_string(), user_agent());
        if let Some(auth) = auth {
            for (k, v) in auth {
                headers.insert(k.clone(), v.clone());
            }
        }
        headers
    }

    /// Run the refresh hook (the explicit one, or [`Self::refresh_credentials`]).
    fn run_refresh(&self) -> bool {
        match &self.refresh_fn {
            Some(refresh) => refresh(),
            None => self.refresh_credentials(),
        }
    }

    /// Send an authenticated request, redeeming the refresh token once on a 401.
    ///
    /// An access token expiring mid-session is the ordinary case, not an
    /// error: the user has a refresh token on disk and nothing about their
    /// session has actually ended.
    ///
    /// Keyed on the status, not the error, so it stays clear of the
    /// permission-error hierarchy: a 403 is a verdict about this account's
    /// plan ([`ModelAccessError`]), and re-minting the same identity cannot
    /// change it.  Only a 401 says the token itself is stale.
    ///
    /// *send* is called with the headers to use rather than being handed a
    /// prepared request, so the retry re-reads `auth_headers_fn` and picks up
    /// whatever the refresh left behind.
    ///
    /// The retry is keyed on the credentials having actually *changed*, not on
    /// what the refresh reported.  The refresh can fail and still leave us
    /// better off (redeeming a dead refresh token clears both OAuth
    /// credentials, after which `auth_headers` falls back to a working
    /// developer token), and it can report success without changing what we
    /// send (in which case resending is a guaranteed second 401).  Comparing
    /// credentials also means the hook may decline entirely — as it does in a
    /// per-project agent child — and that a process that never refreshes still
    /// recovers a token another process rotated during the round trip.
    pub fn authed<F, R>(&self, send: F) -> Result<R, reqwest::Error>
    where
        F: Fn(&Headers) -> Result<R, reqwest::Error>,
        R: AuthedResponse,
    {
        let sent = (self.auth_headers_fn)();
        let resp = send(&self.headers(Some(&sent)))?;
        if resp.status_code() != 401 {
            return Ok(resp);
        }

        self.run_refresh();
        let current = (self.auth_headers_fn)();
        if current == sent {
            return Ok(resp);
        }
        send(&self.headers(Some(&current)))
    }

    /// A blocking HTTP client with *timeout* seconds of budget.
    fn http(&self, timeout: u64) -> reqwest::Result<reqwest::blocking::Client> {
        let budget = Duration::from_secs(timeout);
        reqwest::blocking::Client::builder()
            .timeout(budget)
            .connect_timeout(budget)
            .build()
    }

    /// Send one request and read its body.
    fn send(
        &self,
        method: Method,
        path: &str,
        headers: &Headers,
        params: &[(&str, String)],
        json: Option<&Value>,
        timeout: u64,
    ) -> reqwest::Result<HttpResponse> {
        let url = if params.is_empty() {
            self.url(path)
        } else {
            match url::Url::parse_with_params(&self.url(path), params) {
                Ok(url) => url.to_string(),
                // An unparsable base URL: let reqwest report it as a builder
                // error on the raw string.
                Err(_) => self.url(path),
            }
        };
        let mut request = self.http(timeout)?.request(method, url);
        for (name, value) in headers {
            request = request.header(name.as_str(), value.as_str());
        }
        if let Some(body) = json {
            request = request.json(body);
        }
        let response = request.send()?;
        let status = response.status().as_u16();
        let body = response.bytes()?;
        Ok(HttpResponse { status, body })
    }

    /// GET /v1/bootstrap/models (no auth). Returns the array of model entries.
    pub fn list_bootstrap_models(&self) -> PlatformResult<Vec<JsonObject>> {
        let resp = self
            .send(Method::GET, BOOTSTRAP_MODELS, &self.headers(None), &[], None, TIMEOUT)
            .map_err(|e| connection_error(&e))?;

        raise_for_status(
            resp.status,
            &[(404, "Bootstrap endpoint not found on server.".to_string())],
            ErrorClass::Connection,
        )?;

        let data = resp.json()?;
        if !truthy(&data) {
            return Err(PlatformError::Value("Bootstrap API returned no models".to_string()));
        }
        object_list(data)
    }

    /// GET /v1/models?types=all (auth optional). Returns list of model entries.
    ///
    /// `types=all` asks for both entries of a model listed as remote and as
    /// local under one name (see [`ModelRegistry::hydrate`](crate::model::spec::ModelRegistry::hydrate));
    /// without it the API sends one row per name for older clients.
    pub fn list_models(&self) -> PlatformResult<Vec<JsonObject>> {
        let params = [("types", "all".to_string())];
        let mut resp = self
            .authed(|h| self.send(Method::GET, MODELS, h, &params, None, TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        if resp.status == 401 {
            // Still refused after `authed` tried the refresh token, so the
            // credentials are genuinely dead.  The catalog is public, so ask
            // again anonymously rather than showing an empty picker — the
            // entries come back with `is_accessible` false, which is honest.
            resp = self
                .send(Method::GET, MODELS, &self.headers(None), &params, None, TIMEOUT)
                .map_err(|e| connection_error(&e))?;
        }

        raise_for_status(resp.status, &[], ErrorClass::Connection)?;
        object_list(resp.json()?)
    }

    /// GET /v1/releases/latest?channel=... (no auth). Returns the
    /// `LatestReleaseResponse` wrapper `{update_available, update_required, release}`.
    ///
    /// We poll WITHOUT `current_version` so the cached payload is
    /// client-agnostic; `update_available`/`update_required` are computed
    /// locally per request.  Callers use the `release` field (which may be
    /// `null` when no published release exists for the channel).
    pub fn get_latest_release(&self, channel: &str) -> PlatformResult<Option<JsonObject>> {
        let params = [("channel", channel.to_string())];
        let resp = self
            .send(Method::GET, RELEASES_LATEST, &self.headers(None), &params, None, TIMEOUT)
            .map_err(|e| connection_error(&e))?;

        raise_for_status(resp.status, &[], ErrorClass::Connection)?;
        match resp.json()? {
            Value::Null => Ok(None),
            other => object(other).map(Some),
        }
    }

    /// GET /v1/mcp-catalog (auth required). Curated MCP server catalog entries.
    pub fn list_mcp_catalog(&self) -> PlatformResult<Vec<JsonObject>> {
        let resp = self
            .authed(|h| self.send(Method::GET, MCP_CATALOG, h, &[], None, TIMEOUT))
            .map_err(|e| {
                // Only a failure to connect is a `ConnectionError` here (Python
                // catches `httpx.ConnectError` alone); any other transport
                // failure surfaces as itself.
                if e.is_connect() {
                    connection_error(&e)
                } else {
                    PlatformError::Other(e.to_string())
                }
            })?;

        raise_for_status(
            resp.status,
            &[(401, "Authentication failed. Check your SLP_DEVELOPER_TOKEN.".to_string())],
            ErrorClass::Permission,
        )?;
        object_list(resp.json()?)
    }

    /// GET /v1/models/{model_name}/download (auth optional).
    ///
    /// Returns the `{filename: signed_url}` map in server order.
    ///
    /// Auth headers are sent when we have them, but their absence is not an
    /// error: the platform serves free models to anonymous callers and
    /// answers 403 for anything the caller's plan doesn't grant.  That 403 is
    /// raised as [`ModelAccessError`] so callers can tell "not on your plan"
    /// apart from the 401 that means the token itself is bad.
    ///
    /// Failures are split by whether retrying could change the answer.
    /// `ConnectionError` covers the transient ones (429, 5xx): the download
    /// route is rate-limited per IP and anonymous downloads now reach that
    /// limit, so a 429 means "wait a moment", not "upgrade your plan".
    /// `PermissionError` covers the verdicts about this caller and this model
    /// (401/403/404), which no amount of retrying fixes.  Every one of them
    /// carries `status_code`.  Empty artifacts → `ValueError`.
    pub fn get_model_artifacts(&self, model_name: &str) -> PlatformResult<Vec<(String, String)>> {
        let path = MODEL_DOWNLOAD.replacen("{}", model_name, 1);
        let resp = self
            .authed(|h| self.send(Method::GET, &path, h, &[], None, TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        if resp.status == 403 {
            return Err(ModelAccessError::new(format!(
                "Model '{model_name}' is not available on your current plan. \
                 Please upgrade to access this model."
            ))
            .into());
        }

        raise_for_status(
            resp.status,
            &[
                // Reaches an end user in the app, so it says what they can do
                // about it rather than naming an env var they have never set.
                (401, "Sign in to download this model.".to_string()),
                (404, format!("Model '{model_name}' not found on server.")),
                (429, "Too many download requests. Please wait a moment and try again.".to_string()),
                (500, format!("Server error while fetching model '{model_name}' artifacts.")),
            ],
            ErrorClass::Permission,
        )?;

        let data = object(resp.json()?)?;
        let artifacts = data.get("artifacts").cloned().unwrap_or(Value::Object(JsonObject::new()));
        if !truthy(&artifacts) {
            return Err(PlatformError::Value(format!(
                "API returned no artifacts for model '{model_name}'"
            )));
        }
        match artifacts {
            Value::Object(map) => Ok(map
                .into_iter()
                .map(|(name, url)| {
                    let url = match url {
                        Value::String(s) => s,
                        other => other.to_string(),
                    };
                    (name, url)
                })
                .collect()),
            other => Err(PlatformError::Value(format!(
                "API returned malformed artifacts for model '{model_name}': {other}"
            ))),
        }
    }

    /// GET /v1/users/me (auth required). Returns user profile dict.
    pub fn get_current_user(&self) -> PlatformResult<JsonObject> {
        let resp = self
            .authed(|h| self.send(Method::GET, USERS_ME, h, &[], None, TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        raise_for_status(
            resp.status,
            &[(401, "Authentication failed. Token may be expired or invalid.".to_string())],
            ErrorClass::Permission,
        )?;
        object(resp.json()?)
    }

    /// PATCH /v1/users/me (auth required). Returns updated user profile dict.
    pub fn update_current_user(&self, payload: &JsonObject) -> PlatformResult<JsonObject> {
        let body = Value::Object(payload.clone());
        let resp = self
            .authed(|h| self.send(Method::PATCH, USERS_ME, h, &[], Some(&body), TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        raise_for_status(
            resp.status,
            &[(401, "Authentication failed. Token may be expired or invalid.".to_string())],
            ErrorClass::Permission,
        )?;
        object(resp.json()?)
    }

    /// DELETE /v1/users/me (auth required). Permanently deletes the account.
    pub fn delete_current_user(&self) -> PlatformResult<()> {
        let auth = (self.auth_headers_fn)();
        let resp = self
            .send(Method::DELETE, USERS_ME, &self.headers(Some(&auth)), &[], None, TIMEOUT)
            .map_err(|e| connection_error(&e))?;

        raise_for_status(
            resp.status,
            &[(401, "Authentication failed. Token may be expired or invalid.".to_string())],
            ErrorClass::Permission,
        )
    }

    /// POST /v1/auth/jwt/refresh with a refresh token (no auth).
    ///
    /// Returns `access_token` and optionally `refresh_token`; a reply without
    /// `access_token` → `ValueError`.
    pub fn refresh_access_token(&self, refresh_token: &str) -> PlatformResult<JsonObject> {
        let body = serde_json::json!({ "refresh_token": refresh_token });
        let resp = self
            .send(Method::POST, JWT_REFRESH, &self.headers(None), &[], Some(&body), REFRESH_TIMEOUT)
            .map_err(|e| connection_error(&e))?;

        raise_for_status(
            resp.status,
            &[(401, "Authentication failed. Refresh token may be expired or revoked.".to_string())],
            ErrorClass::Permission,
        )?;

        let data = object(resp.json()?)?;
        if !data.get("access_token").map(truthy).unwrap_or(false) {
            return Err(PlatformError::Value("Refresh response missing access_token".to_string()));
        }
        Ok(data)
    }

    /// POST /v1/auth/device/code (no auth): start a device code sign-in.
    ///
    /// Returns `device_code`, `user_code`, `verification_uri`,
    /// `verification_uri_complete`, `expires_in` and `interval`.  An unknown
    /// *client_id* is a `PermissionError` carrying the platform's
    /// `error_description`.
    pub fn request_device_code(&self, client_id: &str) -> PlatformResult<JsonObject> {
        let body = serde_json::json!({ "client_id": client_id });
        let resp = self
            .send(Method::POST, DEVICE_CODE, &self.headers(None), &[], Some(&body), TIMEOUT)
            .map_err(|e| connection_error(&e))?;

        if !is_transient(resp.status) && !(200..300).contains(&resp.status) {
            if let Some((_, description)) = oauth_error(&resp.body) {
                return Err(PlatformError::Permission { message: description, status_code: Some(resp.status) });
            }
        }
        raise_for_status(resp.status, &[], ErrorClass::Permission)?;
        object(resp.json()?)
    }

    /// POST /v1/auth/device/token (no auth): one poll of a device code sign-in.
    ///
    /// The platform answers every state short of approval with a 400 and an
    /// OAuth `error`; the four that are part of the flow come back as
    /// [`DevicePoll`] values.  Any other 400 (an unknown or used code) is a
    /// `PermissionError` with the platform's `error_description`, and a 429 or
    /// 5xx is a `ConnectionError` as everywhere else — worth polling through.
    pub fn poll_device_token(&self, device_code: &str, client_id: &str) -> PlatformResult<DevicePoll> {
        let body = serde_json::json!({
            "grant_type": DEVICE_GRANT_TYPE,
            "device_code": device_code,
            "client_id": client_id,
        });
        let resp = self
            .send(Method::POST, DEVICE_TOKEN, &self.headers(None), &[], Some(&body), REFRESH_TIMEOUT)
            .map_err(|e| connection_error(&e))?;

        if resp.status == 400 {
            if let Some((error, description)) = oauth_error(&resp.body) {
                return match error.as_str() {
                    "authorization_pending" => Ok(DevicePoll::Pending),
                    "slow_down" => Ok(DevicePoll::SlowDown),
                    "access_denied" => Ok(DevicePoll::Denied),
                    "expired_token" => Ok(DevicePoll::Expired),
                    _ => Err(PlatformError::Permission { message: description, status_code: Some(400) }),
                };
            }
        }
        raise_for_status(resp.status, &[], ErrorClass::Permission)?;

        let data = object(resp.json()?)?;
        if !data.get("access_token").map(truthy).unwrap_or(false) {
            return Err(PlatformError::Value("Device token response missing access_token".to_string()));
        }
        Ok(DevicePoll::Tokens(data))
    }

    /// POST /v1/models/{name}/upload (auth required). Returns `{url, blob_path, expires_at}`.
    pub fn get_model_upload_url(&self, model_name: &str, filename: &str) -> PlatformResult<JsonObject> {
        let path = MODEL_UPLOAD.replacen("{}", model_name, 1);
        let body = serde_json::json!({ "filename": filename });
        let resp = self
            .authed(|h| self.send(Method::POST, &path, h, &[], Some(&body), TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        raise_for_status(
            resp.status,
            &[
                (401, "Authentication failed. Check your SLP_DEVELOPER_TOKEN.".to_string()),
                (404, "Model upload endpoint not found on server.".to_string()),
            ],
            ErrorClass::Permission,
        )?;
        object(resp.json()?)
    }

    /// POST /v1/usage/tokens (auth required). Fire-and-forget telemetry;
    /// failures are logged, never raised.
    pub fn push_token_usage(&self, record: &JsonObject, timeout: u64) {
        // Deliberately not through `authed`.  A 401 here would redeem the
        // refresh token, and a refresh endpoint answering 401 clears both
        // credentials — so a fire-and-forget telemetry POST, whose every
        // failure is swallowed below, could log the user out with no visible
        // cause.  Usage records are flushed in batches, so it would do it once
        // per queued record.
        let auth = (self.auth_headers_fn)();
        let body = Value::Object(record.clone());
        let result = self
            .send(Method::POST, USAGE_TOKENS, &self.headers(Some(&auth)), &[], Some(&body), timeout)
            .map_err(|e| connection_error(&e))
            .and_then(|resp| raise_for_status(resp.status, &[], ErrorClass::Connection));
        if let Err(e) = result {
            tracing::error!(target: "smartloop.platform.client", "Failed to push token usage: {}", e);
        }
    }

    /// GET /v1/websearch?q=..&count=.. (auth required). Returns the result URLs.
    ///
    /// The endpoint resolves the account's registered search provider
    /// server-side, so this side never sees a provider name or an API key.
    /// A 401/403 means search is unavailable for the account, and a 429 means
    /// its monthly allowance is spent; both raise `PermissionError`, because
    /// both mean "stop searching" rather than "try again".  Every other
    /// failure is a `ConnectionError`, which callers must be free to treat as
    /// transient.
    pub fn web_search(&self, query: &str, count: u32, timeout: u64) -> PlatformResult<Vec<String>> {
        let params = [("q", query.to_string()), ("count", count.to_string())];
        let resp = self
            .authed(|h| self.send(Method::GET, WEBSEARCH, h, &params, None, timeout))
            .map_err(|e| connection_error(&e))?;

        if matches!(resp.status, 401 | 403 | 429) {
            return Err(PlatformError::permission(format!(
                "Web search endpoint rejected the token (HTTP {})",
                resp.status
            )));
        }
        raise_for_status(resp.status, &[], ErrorClass::Connection)?;

        match resp.json()? {
            Value::Array(items) => Ok(items
                .into_iter()
                .filter_map(|item| match item {
                    Value::String(s) if !s.trim().is_empty() => Some(s),
                    _ => None,
                })
                .collect()),
            _ => Ok(Vec::new()),
        }
    }

    /// GET /v1/websearch/status (auth required). Whether search would serve.
    ///
    /// Answers for the whole account: signed in, and with searches left in the
    /// month's allowance.  Raises `PermissionError` on 401/403 and
    /// `ConnectionError` on anything else, keeping "search is off"
    /// distinguishable from "the check did not complete" — see
    /// `search::providers::hosted_search_available`, which fails open on the
    /// latter.
    pub fn web_search_status(&self, timeout: u64) -> PlatformResult<bool> {
        let status = self.web_search_status_detail(timeout)?;
        Ok(status.get("enabled").map(truthy).unwrap_or(false))
    }

    /// GET /v1/websearch/status (auth required), the whole body: `enabled`,
    /// `reason`, and the account's monthly `quota` (limit, used, remaining,
    /// resets_at, plan).  Same errors as [`Self::web_search_status`].
    pub fn web_search_status_detail(&self, timeout: u64) -> PlatformResult<JsonObject> {
        let resp = self
            .authed(|h| self.send(Method::GET, WEBSEARCH_STATUS, h, &[], None, timeout))
            .map_err(|e| connection_error(&e))?;

        if matches!(resp.status, 401 | 403) {
            return Err(PlatformError::permission(format!(
                "Web search status rejected the token (HTTP {})",
                resp.status
            )));
        }
        raise_for_status(resp.status, &[], ErrorClass::Connection)?;
        match resp.json()? {
            Value::Object(map) => Ok(map),
            other => Err(PlatformError::Other(format!(
                "Unexpected web search status response: {other}"
            ))),
        }
    }

    /// GET /v1/chat/quota (auth required). The account's cloud model
    /// allowance this period: `percent_used`, `percent_available`,
    /// `resets_at`, `plan`, request and token counts.  Credit amounts
    /// (`limit`, `used`, `remaining`) are null except for superusers.
    ///
    /// 401 and 403 raise `PermissionError` carrying the status: a 401 means
    /// sign in, a 403 that the account has no cloud access.  Anything else is
    /// `ConnectionError`.
    pub fn cloud_quota(&self, timeout: u64) -> PlatformResult<JsonObject> {
        let resp = self
            .authed(|h| self.send(Method::GET, CHAT_QUOTA, h, &[], None, timeout))
            .map_err(|e| connection_error(&e))?;
        raise_for_status(resp.status, &[], ErrorClass::Permission)?;
        object(resp.json()?)
    }

    /// POST /v1/payments/apple/verify (auth required).
    ///
    /// Hands a Mac App Store transaction id to the platform, which checks it
    /// with Apple and links the subscription to this account.  Returns
    /// `{is_paid, subscription_status, current_period_end}`.
    ///
    /// Every refusal carries `status_code`, and — except a 401, which always
    /// means "sign in" — the platform's own `detail` as its message: a 400
    /// (invalid transaction) or 409 (linked to another account) is a
    /// `PermissionError`, a verdict retrying will not change; a 503 (Apple IAP
    /// not configured) or any other 5xx/429 is a `ConnectionError`, as
    /// [`raise_for_status`] classifies every transient status.
    pub fn verify_apple_transaction(&self, transaction_id: &str) -> PlatformResult<JsonObject> {
        let body = serde_json::json!({ "transaction_id": transaction_id });
        let resp = self
            .authed(|h| self.send(Method::POST, APPLE_VERIFY, h, &[], Some(&body), TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        raise_relaying_detail(&resp)?;
        object(resp.json()?)
    }

    /// POST /v1/payments/create-checkout-session (auth required, body `{}`).
    ///
    /// Starts a Stripe Checkout for the Pro upgrade on builds that are not
    /// distributed through the Mac App Store.  Returns `{checkout_url}`.
    ///
    /// Errors as [`Self::verify_apple_transaction`]: 401 is "sign in"; a 404
    /// (Stripe disabled on the platform) or 400 (a Stripe error) is a
    /// `PermissionError` carrying the platform's `detail` and status.
    pub fn create_checkout_session(&self) -> PlatformResult<JsonObject> {
        // An empty JSON object rather than no body: Google's load balancer in
        // front of the platform answers a bodiless POST with 411 Length
        // Required, which surfaced in the app as "API error (411)".
        let body = serde_json::json!({});
        let resp = self
            .authed(|h| self.send(Method::POST, CHECKOUT_SESSION, h, &[], Some(&body), TIMEOUT))
            .map_err(|e| connection_error(&e))?;

        raise_relaying_detail(&resp)?;
        object(resp.json()?)
    }
}

/// [`raise_for_status`] for the payment endpoints, whose refusals are written
/// for the user: a 401 says "sign in", any other non-2xx carries the
/// platform's `detail` (when it sent one) as its message.  Refusals are
/// `PermissionError`; transient statuses stay `ConnectionError`.
fn raise_relaying_detail(resp: &HttpResponse) -> PlatformResult<()> {
    let mut error_map = vec![(401, "Authentication failed. Token may be expired or invalid.".to_string())];
    if resp.status != 401 {
        if let Some(detail) = error_detail(&resp.body) {
            error_map.push((resp.status, detail));
        }
    }
    raise_for_status(resp.status, &error_map, ErrorClass::Permission)
}

/// A response [`SLPPlatformClient::authed`] can inspect for a 401.
pub trait AuthedResponse {
    /// The HTTP status code.
    fn status_code(&self) -> u16;
}

impl AuthedResponse for HttpResponse {
    fn status_code(&self) -> u16 {
        self.status
    }
}

impl AuthedResponse for reqwest::blocking::Response {
    fn status_code(&self) -> u16 {
        self.status().as_u16()
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
    fn web_search_status_detail(&self, timeout: u64) -> PlatformResult<JsonObject> {
        SLPPlatformClient::web_search_status_detail(self, timeout)
    }
    fn cloud_quota(&self, timeout: u64) -> PlatformResult<JsonObject> {
        SLPPlatformClient::cloud_quota(self, timeout)
    }
    fn verify_apple_transaction(&self, transaction_id: &str) -> PlatformResult<JsonObject> {
        SLPPlatformClient::verify_apple_transaction(self, transaction_id)
    }
    fn create_checkout_session(&self) -> PlatformResult<JsonObject> {
        SLPPlatformClient::create_checkout_session(self)
    }
    fn request_device_code(&self, client_id: &str) -> PlatformResult<JsonObject> {
        SLPPlatformClient::request_device_code(self, client_id)
    }
    fn poll_device_token(&self, device_code: &str, client_id: &str) -> PlatformResult<DevicePoll> {
        SLPPlatformClient::poll_device_token(self, device_code, client_id)
    }
}
