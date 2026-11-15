//! Sync MCP client service for connecting to remote MCP servers and discovering
//! tools (JSON-RPC 2.0 over streamable HTTP, hand-rolled on blocking reqwest).
//!
//! Port of `smartloop/mcp/client.py`.  Blocking: call from a plain thread or
//! `tokio::task::spawn_blocking`.
//!
//! The HTTP layer is reached through the [`HttpTransport`] trait (the
//! `httpx.Client` stand-in), shared with [`crate::mcp::oauth`].  The default is
//! a blocking reqwest client; [`set_http_transport_override`] swaps every
//! client this module and the OAuth service open for a fake — the analogue of
//! the Python tests' `patch("httpx.Client", ...)`.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use once_cell::sync::Lazy;
use parking_lot::RwLock;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::mcp::session_pool::PooledSession;
use crate::workspace::{MCPToolAnnotations, MCPToolDefinition};

/// Python logger name of this module.
const LOGGER: &str = "smartloop.mcp.client";

// ------------------------------------------------------------------ errors

/// Connection-level failure (HTTP status, transport).  Carries the HTTP
/// status, and the `WWW-Authenticate` header on a 401.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct MCPConnectionError {
    pub message: String,
    pub www_authenticate: Option<String>,
    pub status_code: Option<u16>,
    /// The server no longer knows the `Mcp-Session-Id` the request carried.
    pub session_lost: bool,
}

impl MCPConnectionError {
    pub fn new(message: impl Into<String>, www_authenticate: Option<String>, status_code: Option<u16>) -> Self {
        Self { message: message.into(), www_authenticate, status_code, session_lost: false }
    }
}

/// A JSON-RPC level error or unparseable response.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MCPProtocolError(pub String);

/// Raised when the MCP subprocess fails to start or exits before the handshake.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MCPLaunchError(pub String);

/// Raised when an MCP server is reachable-ish but its tools cannot be
/// discovered (launcher-not-found, handshake failures, `tools/list` errors).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MCPDiscoveryError(pub String);

/// Discovery failed because the connection can no longer authenticate (a
/// specialization of [`MCPDiscoveryError`]: the refresh was attempted and the
/// provider refused the grant; the connection is already marked).
///
/// Separate from its parent because the two want different reporting: this
/// one has already been diagnosed, and raising the generic error would have
/// the "reconnect it" sentence overwritten by a transport message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MCPAuthExpiredError(pub String);

/// The `MCPClientError` hierarchy as one enum.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MCPClientError {
    #[error(transparent)]
    Connection(#[from] MCPConnectionError),
    #[error(transparent)]
    Protocol(#[from] MCPProtocolError),
    #[error(transparent)]
    Launch(#[from] MCPLaunchError),
    #[error(transparent)]
    Discovery(#[from] MCPDiscoveryError),
    #[error(transparent)]
    AuthExpired(#[from] MCPAuthExpiredError),
    /// A bare `MCPClientError(message)`, or any other failure (the Python
    /// code's generic `Exception` paths).
    #[error("{0}")]
    Other(String),
}

impl MCPClientError {
    /// True for `MCPDiscoveryError` and its subclass `MCPAuthExpiredError`.
    pub fn is_discovery(&self) -> bool {
        matches!(self, Self::Discovery(_) | Self::AuthExpired(_))
    }

    /// The HTTP status a connection error carries, if any.
    pub fn status_code(&self) -> Option<u16> {
        match self {
            Self::Connection(e) => e.status_code,
            _ => None,
        }
    }

    /// True when the server refused the request's session as unknown or
    /// expired, so a fresh `initialize` can recover it.
    pub fn is_session_lost(&self) -> bool {
        matches!(self, Self::Connection(e) if e.session_lost)
    }
}

// ------------------------------------------------------------------ HTTP layer

/// A response as the MCP/OAuth code needs it: status, headers, body text.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HttpResponse {
    pub status: u16,
    /// Header `(name, value)` pairs; names are matched case-insensitively.
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl HttpResponse {
    /// A response with `status` and `body` and no headers.
    pub fn new(status: u16, body: impl Into<String>) -> Self {
        Self { status, headers: Vec::new(), body: body.into() }
    }

    /// A `200` whose body is `value` serialized as JSON, with a JSON content type.
    pub fn json(status: u16, value: &Value) -> Self {
        Self {
            status,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: value.to_string(),
        }
    }

    /// Add a header (builder style).
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// The first value of header `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// `httpx.Response.is_success`: a 2xx status.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// A request body.
#[derive(Debug, Clone, PartialEq)]
pub enum HttpBody {
    /// `json=...`
    Json(Value),
    /// `data=...` (form-encoded), in insertion order.
    Form(Vec<(String, String)>),
}

impl HttpBody {
    /// The form fields as a map (for tests asserting on what was sent).
    pub fn form_map(&self) -> Map<String, Value> {
        match self {
            HttpBody::Form(pairs) => pairs
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect(),
            HttpBody::Json(Value::Object(map)) => map.clone(),
            HttpBody::Json(_) => Map::new(),
        }
    }
}

/// The `httpx.Client` surface the MCP client and OAuth service use.
///
/// `Err` is a transport failure (`httpx.RequestError`): connection refused,
/// DNS, timeout.  Any HTTP status, success or not, is an `Ok` response.
pub trait HttpTransport: Send + Sync {
    fn get(&self, url: &str) -> Result<HttpResponse, String>;
    fn post(&self, url: &str, body: &HttpBody, headers: &[(String, String)]) -> Result<HttpResponse, String>;
}

/// The real transport: a blocking reqwest client.
///
/// Redirects are not followed, matching httpx's default — a 3xx is a status
/// the caller sees, not a hop taken silently.
pub struct ReqwestTransport {
    client: reqwest::blocking::Client,
}

impl ReqwestTransport {
    /// A client with the given total timeout in seconds.
    pub fn new(timeout_secs: f64) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(timeout_secs))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self { client }
    }

    /// A client whose idle connections stay open for `idle_secs`, for reuse
    /// across calls.
    pub fn pooled(timeout_secs: f64, idle_secs: u64) -> Self {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs_f64(timeout_secs))
            .redirect(reqwest::redirect::Policy::none())
            .pool_idle_timeout(Duration::from_secs(idle_secs))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());
        Self { client }
    }

    /// Wrap an existing client as-is.
    pub fn from_client(client: reqwest::blocking::Client) -> Self {
        Self { client }
    }

    fn finish(resp: reqwest::blocking::Response) -> Result<HttpResponse, String> {
        let status = resp.status().as_u16();
        let headers = resp
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), String::from_utf8_lossy(v.as_bytes()).into_owned()))
            .collect();
        let body = resp.text().map_err(|e| e.to_string())?;
        Ok(HttpResponse { status, headers, body })
    }
}

impl HttpTransport for ReqwestTransport {
    fn get(&self, url: &str) -> Result<HttpResponse, String> {
        let resp = self.client.get(url).send().map_err(|e| e.to_string())?;
        Self::finish(resp)
    }

    fn post(&self, url: &str, body: &HttpBody, headers: &[(String, String)]) -> Result<HttpResponse, String> {
        let mut req = self.client.post(url);
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        let has_ct = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-type"));
        req = match body {
            HttpBody::Json(value) => {
                let req = req.body(value.to_string());
                if has_ct { req } else { req.header("Content-Type", "application/json") }
            }
            HttpBody::Form(pairs) => {
                let encoded: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
                let req = req.body(crate::mcp::oauth::urlencode(&encoded));
                if has_ct { req } else { req.header("Content-Type", "application/x-www-form-urlencoded") }
            }
        };
        let resp = req.send().map_err(|e| e.to_string())?;
        Self::finish(resp)
    }
}

static TRANSPORT_OVERRIDE: Lazy<RwLock<Option<Arc<dyn HttpTransport>>>> = Lazy::new(|| RwLock::new(None));

/// Replace (or with `None`, restore) the transport every newly opened
/// [`MCPClient`] / owned [`crate::mcp::oauth::MCPOAuthService`] uses.
/// Returns the previous override.
///
/// The analogue of `patch("httpx.Client", return_value=fake)`: process-wide,
/// so tests that use it must be serialized.  Pooled sessions are dropped, since
/// they were opened over the transport being replaced.
pub fn set_http_transport_override(transport: Option<Arc<dyn HttpTransport>>) -> Option<Arc<dyn HttpTransport>> {
    crate::mcp::session_pool::reset_mcp_sessions();
    std::mem::replace(&mut *TRANSPORT_OVERRIDE.write(), transport)
}

/// The transport a newly opened client uses: the override when one is set,
/// else a fresh blocking reqwest client with `timeout_secs`.
pub fn default_http_transport(timeout_secs: f64) -> Arc<dyn HttpTransport> {
    if let Some(t) = TRANSPORT_OVERRIDE.read().clone() {
        return t;
    }
    Arc::new(ReqwestTransport::new(timeout_secs))
}

/// The process-wide MCP client, so tool calls reuse TCP/TLS connections.
/// Auth travels per request, so one client serves every server.
static SHARED_TRANSPORT: Lazy<Arc<dyn HttpTransport>> = Lazy::new(|| Arc::new(ReqwestTransport::pooled(60.0, 300)));

/// The transport [`MCPClient::enter`] uses: the override when one is set,
/// else the shared pooled client.
pub fn shared_http_transport() -> Arc<dyn HttpTransport> {
    if let Some(t) = TRANSPORT_OVERRIDE.read().clone() {
        return t;
    }
    SHARED_TRANSPORT.clone()
}

/// RAII guard installing a transport override for its lifetime.
pub struct HttpTransportOverride {
    previous: Option<Arc<dyn HttpTransport>>,
}

impl HttpTransportOverride {
    pub fn install(transport: Arc<dyn HttpTransport>) -> Self {
        Self { previous: set_http_transport_override(Some(transport)) }
    }
}

impl Drop for HttpTransportOverride {
    fn drop(&mut self) {
        set_http_transport_override(self.previous.take());
    }
}

// ------------------------------------------------------------------ helpers

/// Python `str()` of a JSON value as these modules format it into messages:
/// strings raw, `null` as `None`, booleans as `True`/`False`.
pub(crate) fn py_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    }
}

/// Python truthiness of a JSON value.
pub(crate) fn py_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// The first `n` characters of `s` (Python `s[:n]`).
pub(crate) fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// An optional string field of a JSON object (non-strings are stringified;
/// `null`/missing is `None`).
pub(crate) fn opt_string(obj: &Map<String, Value>, key: &str) -> Option<String> {
    match obj.get(key) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => Some(py_str(Some(other))),
    }
}

/// Build an `MCPServerInfo` from an `initialize` result's `serverInfo`, or
/// `None` when it is missing or empty.
pub(crate) fn server_info_from(init_result: &Map<String, Value>) -> Result<Option<MCPServerInfo>, MCPClientError> {
    match init_result.get("serverInfo") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(raw)) if raw.is_empty() => Ok(None),
        Some(Value::Object(raw)) => Ok(Some(MCPServerInfo {
            name: opt_string(raw, "name"),
            title: opt_string(raw, "title"),
            version: opt_string(raw, "version"),
            description: opt_string(raw, "description"),
            icons: match raw.get("icons") {
                Some(Value::Array(items)) => Some(items.clone()),
                _ => None,
            },
        })),
        Some(other) if !py_truthy(Some(other)) => Ok(None),
        Some(_) => Err(MCPClientError::Other("'serverInfo' is not an object".to_string())),
    }
}

/// Build the tool list from a `tools/list` result.
pub(crate) fn tools_from(tools_result: &Map<String, Value>) -> Result<Vec<MCPToolDefinition>, MCPClientError> {
    let raw_tools = match tools_result.get("tools") {
        None => return Ok(Vec::new()),
        Some(Value::Array(items)) => items,
        Some(_) => return Err(MCPClientError::Other("'tools' is not a list".to_string())),
    };
    let mut tools = Vec::with_capacity(raw_tools.len());
    for tool_data in raw_tools {
        let Some(tool) = tool_data.as_object() else {
            return Err(MCPClientError::Other("tool entry is not an object".to_string()));
        };
        let name = match tool.get("name") {
            None => String::new(),
            Some(Value::String(s)) => s.clone(),
            Some(_) => {
                return Err(MCPClientError::Other(
                    "1 validation error for MCPToolDefinition\nname\n  Input should be a valid string".to_string(),
                ))
            }
        };
        let annotations = tool.get("annotations").and_then(Value::as_object);
        tools.push(MCPToolDefinition {
            name,
            description: opt_string(tool, "description"),
            input_schema: tool.get("inputSchema").and_then(|v| v.as_object().cloned()),
            title: non_empty_str(tool, "title").or_else(|| annotations.and_then(|a| non_empty_str(a, "title"))),
            annotations: annotations.and_then(annotations_from),
        });
    }
    Ok(tools)
}

/// A string field, or `None` when it is missing, not a string, or blank.
fn non_empty_str(obj: &Map<String, Value>, key: &str) -> Option<String> {
    obj.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(str::to_string)
}

/// The behaviour hints of a tool's `annotations`, or `None` when it carries none.
///
/// A hint that is not a boolean is treated as absent.
fn annotations_from(raw: &Map<String, Value>) -> Option<MCPToolAnnotations> {
    let hint = |key: &str| raw.get(key).and_then(Value::as_bool);
    let annotations = MCPToolAnnotations {
        read_only_hint: hint("readOnlyHint"),
        destructive_hint: hint("destructiveHint"),
        idempotent_hint: hint("idempotentHint"),
        open_world_hint: hint("openWorldHint"),
    };
    (annotations != MCPToolAnnotations::default()).then_some(annotations)
}

/// The `content` / `isError` pair of a `tools/call` result.
pub(crate) fn call_result_parts(result: &Map<String, Value>) -> (Option<Vec<Value>>, bool) {
    let content = match result.get("content") {
        None => Some(Vec::new()),
        Some(Value::Null) => None,
        Some(Value::Array(items)) => Some(items.clone()),
        Some(other) => Some(vec![other.clone()]),
    };
    (content, py_truthy(result.get("isError")))
}

/// `f"MCP error {code}: {message}"` for a JSON-RPC error object.
pub(crate) fn jsonrpc_error_message(error: &Value) -> String {
    let (code, message) = match error.as_object() {
        Some(obj) => (py_str(obj.get("code")), py_str(obj.get("message"))),
        None => ("None".to_string(), "None".to_string()),
    };
    format!("MCP error {code}: {message}")
}

// ------------------------------------------------------------------ results

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPServerInfo {
    pub name: Option<String>,
    pub title: Option<String>,
    pub version: Option<String>,
    pub description: Option<String>,
    pub icons: Option<Vec<Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPConnectResult {
    pub success: bool,
    #[serde(default)]
    pub tools: Vec<MCPToolDefinition>,
    pub server_info: Option<MCPServerInfo>,
    pub error: Option<String>,
    /// Resource URI from a 401 `WWW-Authenticate`.
    pub resource_hint: Option<String>,
    /// PRM metadata URL from a 401 `WWW-Authenticate`.
    pub resource_metadata_url: Option<String>,
    /// The server rejected the token (HTTP 401), as on a tool call.  Discovery
    /// runs with the same credentials and ages out the same way, so a caller
    /// re-checking a connection needs to tell "renew and try again" from "this
    /// endpoint is dead".
    #[serde(default)]
    pub auth_expired: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPToolCallResult {
    pub success: bool,
    pub content: Option<Vec<Value>>,
    #[serde(default)]
    pub is_error: bool,
    pub error: Option<String>,
    /// The server rejected the token (HTTP 401), so refreshing it may help.
    /// Callers used to ask `"401" in result.error` instead — a substring test
    /// against a message that embeds the server's own response body.
    #[serde(default)]
    pub auth_expired: bool,
}

// ------------------------------------------------------------------ client

static RESOURCE_QUOTED: Lazy<Regex> = Lazy::new(|| Regex::new(r#"resource="([^"]+)""#).unwrap());
static RESOURCE_BARE: Lazy<Regex> = Lazy::new(|| Regex::new(r"resource=(\S+)").unwrap());
static RESOURCE_METADATA_QUOTED: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"resource_metadata="([^"]+)""#).unwrap());
static RESOURCE_METADATA_BARE: Lazy<Regex> = Lazy::new(|| Regex::new(r"resource_metadata=(\S+)").unwrap());

/// Sync MCP client for connecting to remote MCP servers.
///
/// `enter()` opens the HTTP client (60 s timeout), `exit()` closes it and drops
/// the session (Python's context-manager protocol).
#[derive(Default)]
pub struct MCPClient {
    client: Option<Arc<dyn HttpTransport>>,
    request_id: u64,
    session_id: Option<String>,
    /// The shared session [`Self::enter_pooled`] attached, if any.
    pooled: Option<Arc<PooledSession>>,
    /// Time spent on `initialize` during the current call, if it ran.
    handshake: Option<Duration>,
    /// Time spent waiting for another caller's handshake on a pooled session.
    session_wait: Duration,
}

impl fmt::Debug for MCPClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPClient")
            .field("open", &self.client.is_some())
            .field("request_id", &self.request_id)
            .field("session_id", &self.session_id)
            .field("pooled", &self.pooled.is_some())
            .finish()
    }
}

impl MCPClient {
    pub const PROTOCOL_VERSION: &'static str = "2024-11-05";
    pub const CLIENT_INFO_NAME: &'static str = "smartloop-cli";
    pub const CLIENT_INFO_VERSION: &'static str = "1.0.0";

    pub fn new() -> Self {
        Self::default()
    }

    /// A client already holding `transport` (the tests' `client._client = fake`).
    pub fn with_transport(transport: Arc<dyn HttpTransport>) -> Self {
        let mut client = Self::default();
        client.client = Some(transport);
        client
    }

    /// Replace the HTTP transport in place.
    pub fn set_transport(&mut self, transport: Option<Arc<dyn HttpTransport>>) {
        self.client = transport;
    }

    /// `{"name": "smartloop-cli", "version": "1.0.0"}`.
    pub fn client_info() -> Value {
        serde_json::json!({
            "name": Self::CLIENT_INFO_NAME,
            "version": Self::CLIENT_INFO_VERSION,
        })
    }

    /// Open the HTTP client (`__enter__`).
    pub fn enter(&mut self) -> &mut Self {
        self.client = Some(shared_http_transport());
        self
    }

    /// Open the HTTP client on a session shared with other calls, so the
    /// handshake runs only when the session has not been initialized yet.
    pub fn enter_pooled(&mut self, session: Arc<PooledSession>) -> &mut Self {
        self.client = Some(shared_http_transport());
        self.pooled = Some(session);
        self
    }

    /// Close the HTTP client and forget the session (`__exit__`).  A pooled
    /// session stays in the pool for the next call.
    pub fn exit(&mut self) {
        self.client = None;
        self.session_id = None;
        if let Some(pooled) = self.pooled.take() {
            pooled.touch();
        }
    }

    /// The current `Mcp-Session-Id`, if the server issued one.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    fn next_request_id(&mut self) -> String {
        if let Some(pooled) = &self.pooled {
            return pooled.next_request_id().to_string();
        }
        self.request_id += 1;
        self.request_id.to_string()
    }

    fn get_headers(&self, access_token: Option<&str>, _token_type: &str) -> Vec<(String, String)> {
        let mut headers = vec![
            ("Content-Type".to_string(), "application/json".to_string()),
            ("Accept".to_string(), "application/json, text/event-stream".to_string()),
        ];
        if let Some(token) = access_token.filter(|t| !t.is_empty()) {
            headers.push(("Authorization".to_string(), format!("Bearer {token}")));
        }
        if let Some(session) = &self.session_id {
            headers.push(("Mcp-Session-Id".to_string(), session.clone()));
        }
        headers
    }

    /// Parse the resource URI from a `WWW-Authenticate` header (RFC 6750 / 8707).
    pub fn parse_resource_from_www_authenticate(www_authenticate: &str) -> Option<String> {
        if www_authenticate.is_empty() {
            return None;
        }
        // Match resource="..." or resource=<non-whitespace>
        if let Some(c) = RESOURCE_QUOTED.captures(www_authenticate) {
            return Some(c[1].to_string());
        }
        if let Some(c) = RESOURCE_BARE.captures(www_authenticate) {
            return Some(c[1].trim_end_matches(',').to_string());
        }
        None
    }

    /// Parse the PRM metadata URL from a `WWW-Authenticate` header (RFC 9728 §5.1).
    ///
    /// The server advertises the Protected Resource Metadata location via
    /// `resource_metadata="<url>"`, distinct from the RFC 8707 `resource=`
    /// field parsed by [`Self::parse_resource_from_www_authenticate`].
    pub fn parse_resource_metadata_from_www_authenticate(www_authenticate: &str) -> Option<String> {
        if www_authenticate.is_empty() {
            return None;
        }
        // The literal `_metadata` suffix keeps this from matching the plain
        // `resource=` field.
        if let Some(c) = RESOURCE_METADATA_QUOTED.captures(www_authenticate) {
            return Some(c[1].to_string());
        }
        if let Some(c) = RESOURCE_METADATA_BARE.captures(www_authenticate) {
            return Some(c[1].trim_end_matches(',').to_string());
        }
        None
    }

    /// Extract the JSON-RPC message from an SSE response body.
    pub fn parse_sse_response(&self, response_text: &str) -> Result<Map<String, Value>, MCPClientError> {
        let mut data_lines: Vec<&str> = Vec::new();
        for line in response_text.split('\n') {
            let line = line.trim();
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                let data_content = rest.trim();
                if !data_content.is_empty() {
                    data_lines.push(data_content);
                }
            }
        }

        if !data_lines.is_empty() {
            for data_content in &data_lines {
                if let Ok(Value::Object(result)) = serde_json::from_str::<Value>(data_content) {
                    if result.contains_key("jsonrpc") {
                        return Ok(result);
                    }
                }
            }
            let combined: String = data_lines.concat();
            if let Ok(Value::Object(result)) = serde_json::from_str::<Value>(&combined) {
                return Ok(result);
            }
        }

        Err(MCPProtocolError("No valid JSON found in SSE response".to_string()).into())
    }

    fn transport(&self) -> Result<Arc<dyn HttpTransport>, MCPClientError> {
        self.client
            .clone()
            .ok_or_else(|| MCPClientError::Other("'NoneType' object has no attribute 'post'".to_string()))
    }

    /// POST one JSON-RPC request and return its `result`.
    ///
    /// A 401 becomes an [`MCPConnectionError`] whose message redacts the
    /// credential to its scheme and length.
    pub fn send_request(
        &mut self,
        server_url: &str,
        method: &str,
        params: Option<Value>,
        access_token: Option<&str>,
        token_type: &str,
    ) -> Result<Map<String, Value>, MCPClientError> {
        let mut request = Map::new();
        request.insert("jsonrpc".into(), Value::String("2.0".into()));
        request.insert("id".into(), Value::String(self.next_request_id()));
        request.insert("method".into(), Value::String(method.to_string()));
        if let Some(p) = params.filter(|p| py_truthy(Some(p))) {
            request.insert("params".into(), p);
        }

        let headers = self.get_headers(access_token, token_type);
        let transport = self.transport()?;

        let response = transport
            .post(server_url, &HttpBody::Json(Value::Object(request)), &headers)
            .map_err(|e| MCPConnectionError::new(format!("Connection failed: {e}"), None, None))?;

        if response.status == 401 {
            let www_auth = response.header("www-authenticate").unwrap_or("").to_string();
            // Capture response details for debugging.
            let body = take_chars(&response.body, 500);
            // Not a prefix of the header: "Bearer " is seven characters, so
            // thirty of them is twenty-three characters of a live token — and
            // this string is logged, and becomes `last_error`, which
            // `GET /projects/{id}/mcp` hands to the UI.  What the reader needs
            // is whether a credential was sent, not which one.
            let auth_sent = headers
                .iter()
                .find(|(k, _)| k == "Authorization")
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            let auth_preview = if !auth_sent.is_empty() {
                let (scheme, credential) = match auth_sent.split_once(' ') {
                    Some((s, c)) => (s, c),
                    None => (auth_sent, ""),
                };
                format!("{scheme} ({} chars)", credential.chars().count())
            } else {
                "(none)".to_string()
            };
            let detail = format!(
                "Authentication failed (401). WWW-Authenticate: {}. Response: {}. Authorization sent: {}",
                if www_auth.is_empty() { "(none)" } else { www_auth.as_str() },
                if body.is_empty() { "(empty)" } else { body.as_str() },
                auth_preview,
            );
            return Err(MCPConnectionError::new(detail, Some(www_auth), Some(401)).into());
        }
        if response.status == 403 {
            return Err(MCPConnectionError::new(
                "Authorization failed: Insufficient permissions",
                None,
                Some(403),
            )
            .into());
        }
        if !response.is_success() {
            let sent_session = headers.iter().any(|(k, _)| k == "Mcp-Session-Id");
            let mut error =
                MCPConnectionError::new(format!("HTTP error: {}", response.status), None, Some(response.status));
            error.session_lost = sent_session && session_refused(response.status, &response.body);
            return Err(error.into());
        }

        if let Some(session_id) = response.header("mcp-session-id").filter(|s| !s.is_empty()) {
            self.session_id = Some(session_id.to_string());
        }

        let content_type = response.header("content-type").unwrap_or("");
        let result: Value = if content_type.contains("text/event-stream") {
            Value::Object(self.parse_sse_response(&response.body)?)
        } else {
            match serde_json::from_str::<Value>(&response.body) {
                Ok(v) => v,
                Err(_) => Value::Object(self.parse_sse_response(&response.body)?),
            }
        };

        let Value::Object(result) = result else {
            return Err(MCPClientError::Other(format!(
                "'{}' object has no attribute 'get'",
                json_type_name(&result)
            )));
        };
        if let Some(error) = result.get("error") {
            return Err(MCPProtocolError(jsonrpc_error_message(error)).into());
        }
        match result.get("result") {
            None => Ok(Map::new()),
            Some(Value::Object(r)) => Ok(r.clone()),
            Some(other) => Err(MCPClientError::Other(format!(
                "'{}' object has no attribute 'get'",
                json_type_name(other)
            ))),
        }
    }

    /// POST one JSON-RPC notification; failures are ignored.
    pub fn send_notification(
        &mut self,
        server_url: &str,
        method: &str,
        params: Option<Value>,
        access_token: Option<&str>,
        token_type: &str,
    ) {
        let mut notification = Map::new();
        notification.insert("jsonrpc".into(), Value::String("2.0".into()));
        notification.insert("method".into(), Value::String(method.to_string()));
        if let Some(p) = params.filter(|p| py_truthy(Some(p))) {
            notification.insert("params".into(), p);
        }
        let headers = self.get_headers(access_token, token_type);
        if let Ok(transport) = self.transport() {
            let _ = transport.post(server_url, &HttpBody::Json(Value::Object(notification)), &headers);
        }
    }

    fn initialize_params() -> Value {
        serde_json::json!({
            "protocolVersion": Self::PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": Self::client_info(),
        })
    }

    fn discover(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> Result<MCPConnectResult, MCPClientError> {
        let init_result = self.send_request(
            server_url,
            "initialize",
            Some(Self::initialize_params()),
            access_token,
            token_type,
        )?;
        let server_info = server_info_from(&init_result)?;

        self.send_notification(
            server_url,
            "notifications/initialized",
            Some(Value::Object(Map::new())),
            access_token,
            token_type,
        );

        let tools_result = self.send_request(
            server_url,
            "tools/list",
            Some(Value::Object(Map::new())),
            access_token,
            token_type,
        )?;
        let tools = tools_from(&tools_result)?;

        Ok(MCPConnectResult { success: true, tools, server_info, ..Default::default() })
    }

    /// Initialize, send `notifications/initialized`, then `tools/list`.
    pub fn connect_and_discover_tools(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> MCPConnectResult {
        match self.discover(server_url, access_token, token_type) {
            Ok(result) => result,
            Err(MCPClientError::Connection(e)) => {
                let www = e.www_authenticate.as_deref().filter(|w| !w.is_empty());
                MCPConnectResult {
                    success: false,
                    error: Some(e.to_string()),
                    resource_hint: www.and_then(Self::parse_resource_from_www_authenticate),
                    resource_metadata_url: www.and_then(Self::parse_resource_metadata_from_www_authenticate),
                    auth_expired: e.status_code == Some(401),
                    ..Default::default()
                }
            }
            Err(MCPClientError::Protocol(e)) => {
                MCPConnectResult { success: false, error: Some(e.to_string()), ..Default::default() }
            }
            Err(e) => MCPConnectResult {
                success: false,
                error: Some(format!("Unexpected error: {e}")),
                ..Default::default()
            },
        }
    }

    /// Send `initialize` then `notifications/initialized`.
    fn initialize(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> Result<(), MCPClientError> {
        self.send_request(server_url, "initialize", Some(Self::initialize_params()), access_token, token_type)?;
        self.send_notification(
            server_url,
            "notifications/initialized",
            Some(Value::Object(Map::new())),
            access_token,
            token_type,
        );
        Ok(())
    }

    /// Perform the MCP initialize handshake if no session exists yet.
    ///
    /// A pooled session is initialized once and then reused, whether or not
    /// the server issues a session id.
    fn ensure_session(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> Result<(), MCPClientError> {
        if let Some(pooled) = self.pooled.clone() {
            let waiting = Instant::now();
            let mut state = pooled.state();
            self.session_wait += waiting.elapsed();
            if !state.initialized {
                self.session_id = None;
                self.timed_initialize(server_url, access_token, token_type)?;
                state.session_id = self.session_id.clone();
                state.initialized = true;
            } else {
                self.session_id = state.session_id.clone();
            }
            return Ok(());
        }
        if self.session_id.is_some() {
            return Ok(());
        }
        self.timed_initialize(server_url, access_token, token_type)
    }

    /// [`Self::initialize`], adding its duration to `handshake`.
    fn timed_initialize(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> Result<(), MCPClientError> {
        let started = Instant::now();
        let result = self.initialize(server_url, access_token, token_type);
        self.handshake = Some(self.handshake.unwrap_or_default() + started.elapsed());
        result
    }

    /// How the last call got its session, for the timing log.
    fn session_note(&self, retried: bool) -> String {
        let mut note = match (self.pooled.is_some(), self.handshake) {
            (true, None) => "reused session".to_string(),
            (true, Some(h)) => format!("new session, handshake {} ms", h.as_millis()),
            (false, Some(h)) => format!("unpooled, handshake {} ms", h.as_millis()),
            (false, None) => "unpooled".to_string(),
        };
        if self.session_wait >= Duration::from_millis(10) {
            note.push_str(&format!(", waited {} ms on another call's handshake", self.session_wait.as_millis()));
        }
        if retried {
            note.push_str(", after the server lost the previous session");
        }
        note
    }

    /// Call one tool (`tools/call`).
    pub fn call_tool(
        &mut self,
        server_url: &str,
        tool_name: &str,
        arguments: Option<&Map<String, Value>>,
        access_token: Option<&str>,
        token_type: &str,
    ) -> MCPToolCallResult {
        let started = Instant::now();
        self.handshake = None;
        self.session_wait = Duration::ZERO;
        let params = serde_json::json!({
            "name": tool_name,
            "arguments": Value::Object(arguments.cloned().unwrap_or_default()),
        });
        let mut retried = false;
        let mut outcome = self.ensure_session(server_url, access_token, token_type).and_then(|_| {
            self.send_request(server_url, "tools/call", Some(params.clone()), access_token, token_type)
        });
        // The server never ran the call under a session it no longer knows, so
        // one retry on a fresh session is safe.
        if matches!(&outcome, Err(e) if e.is_session_lost()) {
            tracing::info!(
                target: LOGGER,
                "[mcp-remote] Session on {} expired; initializing a new one",
                server_url
            );
            if let Some(pooled) = &self.pooled {
                pooled.forget(self.session_id.as_deref());
            }
            self.session_id = None;
            retried = true;
            outcome = self.ensure_session(server_url, access_token, token_type).and_then(|_| {
                self.send_request(server_url, "tools/call", Some(params), access_token, token_type)
            });
        }
        tracing::info!(
            target: LOGGER,
            "[mcp-remote] {} on {} took {} ms ({})",
            tool_name,
            server_url,
            started.elapsed().as_millis(),
            self.session_note(retried)
        );
        match outcome {
            Ok(result) => {
                let (content, is_error) = call_result_parts(&result);
                MCPToolCallResult { success: true, content, is_error, ..Default::default() }
            }
            Err(e @ (MCPClientError::Connection(_) | MCPClientError::Protocol(_))) => {
                tracing::error!(
                    target: LOGGER,
                    "[mcp-remote] Tool call '{}' on {} failed: {}",
                    tool_name, server_url, e
                );
                MCPToolCallResult {
                    success: false,
                    error: Some(e.to_string()),
                    auth_expired: e.status_code() == Some(401),
                    ..Default::default()
                }
            }
            Err(e) => {
                tracing::error!(
                    target: LOGGER,
                    "[mcp-remote] Tool call '{}' on {} unexpected error: {}",
                    tool_name, server_url, e
                );
                MCPToolCallResult {
                    success: false,
                    error: Some(format!("Unexpected error: {e}")),
                    ..Default::default()
                }
            }
        }
    }
}

/// A non-2xx answer to a request carrying `Mcp-Session-Id` that means the
/// session is gone: a 404 (the MCP spec's signal), or a 400 saying so.
fn session_refused(status: u16, body: &str) -> bool {
    match status {
        404 => true,
        400 => {
            let body = body.to_ascii_lowercase();
            body.contains("session")
                && (body.contains("not found") || body.contains("expired") || body.contains("invalid"))
        }
        _ => false,
    }
}

/// Python's type name for a JSON value, for error messages.
pub(crate) fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

impl Drop for MCPClient {
    fn drop(&mut self) {
        self.client = None;
        self.session_id = None;
        self.pooled = None;
    }
}
