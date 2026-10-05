//! Sync OAuth service for MCP server authentication (RFC 8414 / 9728 / 7591 /
//! 7636 / 8707 / 9207).
//!
//! Port of `smartloop/mcp/oauth.py`.  Blocking: call from a plain thread or
//! `tokio::task::spawn_blocking`.

use std::fmt;
use std::sync::Arc;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

use crate::mcp::client::{
    default_http_transport, opt_string, py_str, py_truthy, HttpBody, HttpTransport, ReqwestTransport,
};

/// Protected Resource Metadata (RFC 9728).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ProtectedResourceMetadata {
    pub resource: Option<String>,
    #[serde(default)]
    pub authorization_servers: Vec<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
}

/// OAuth 2.0 Authorization Server Metadata.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct OAuthMetadata {
    pub issuer: Option<String>,
    #[serde(default)]
    pub authorization_endpoint: String,
    #[serde(default)]
    pub token_endpoint: String,
    pub registration_endpoint: Option<String>,
    #[serde(default)]
    pub scopes_supported: Vec<String>,
    #[serde(default)]
    pub response_types_supported: Vec<String>,
    #[serde(default)]
    pub grant_types_supported: Vec<String>,
    #[serde(default)]
    pub code_challenge_methods_supported: Vec<String>,
    pub resource: Option<String>,
    /// RFC 9207 §3.  When the server says it sends `iss` on the authorization
    /// response, a response arriving *without* one must be rejected —
    /// otherwise an attacker strips the parameter to get back the behaviour it
    /// defends against.
    #[serde(default)]
    pub authorization_response_iss_parameter_supported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ClientRegistrationResponse {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub client_id_issued_at: Option<i64>,
    pub client_secret_expires_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<i64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
}

// ------------------------------------------------------------------ errors

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MetadataDiscoveryError(pub String);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ResourceMetadataDiscoveryError(pub String);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ClientRegistrationError(pub String);

/// A token endpoint refused a grant, or could not be reached.
///
/// `error_code` is the OAuth 2.0 error identifier from the response body (RFC
/// 6749 §5.2) when the server sent one, and `None` for a transport failure.
/// It is what separates a grant that is dead — `invalid_grant`,
/// `invalid_client` — from a blip: only the first kind means re-minting the
/// same refresh token can never work again, and only the first kind should
/// cost a user their connection.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TokenExchangeError {
    pub message: String,
    pub error_code: Option<String>,
    pub status_code: Option<u16>,
}

impl TokenExchangeError {
    /// Error identifiers that mean the grant itself is finished.  A refresh
    /// token the server has retired, or client credentials it no longer
    /// recognises, cannot be recovered by trying again.
    pub const DEAD_GRANT_CODES: &'static [&'static str] = &["invalid_grant", "invalid_client"];

    pub fn new(message: impl Into<String>, error_code: Option<String>, status_code: Option<u16>) -> Self {
        Self { message: message.into(), error_code, status_code }
    }

    /// True when re-minting the same refresh token can never work again.
    pub fn grant_is_dead(&self) -> bool {
        self.error_code
            .as_deref()
            .map(|code| Self::DEAD_GRANT_CODES.contains(&code))
            .unwrap_or(false)
    }
}

/// The `MCPOAuthError` hierarchy as one enum.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MCPOAuthError {
    #[error(transparent)]
    MetadataDiscovery(#[from] MetadataDiscoveryError),
    #[error(transparent)]
    ResourceMetadataDiscovery(#[from] ResourceMetadataDiscoveryError),
    #[error(transparent)]
    ClientRegistration(#[from] ClientRegistrationError),
    #[error(transparent)]
    TokenExchange(#[from] TokenExchangeError),
    /// A bare `MCPOAuthError(message)`, or any other failure.
    #[error("{0}")]
    Other(String),
}

impl MCPOAuthError {
    /// The [`TokenExchangeError`] inside, if this is one.
    pub fn as_token_exchange(&self) -> Option<&TokenExchangeError> {
        match self {
            Self::TokenExchange(e) => Some(e),
            _ => None,
        }
    }
}

/// Build a [`TokenExchangeError`] from an RFC 6749 §5.2 error body.
///
/// All three fields matter and for different reasons.  `error` is the fixed
/// enum the retry logic switches on (see `DEAD_GRANT_CODES`).
/// `error_description` is the only part that names *what* was wrong, and
/// `error_uri` points at the provider's own page about it.  A body that is not
/// JSON, or not an OAuth error at all, leaves the status line as the whole
/// story rather than failing a second time in here.
///
/// `status` / `body` are the failed response's status and text; `label`
/// prefixes the message (`"{label}: HTTP {status}[: description][ (see uri)]"`).
pub fn token_error(status: u16, body: &str, label: &str) -> TokenExchangeError {
    let mut error_code: Option<String> = None;
    let mut detail = String::new();
    if let Ok(Value::Object(data)) = serde_json::from_str::<Value>(body) {
        error_code = match data.get("error") {
            None | Some(Value::Null) => None,
            Some(v) => Some(py_str(Some(v))),
        };
        let description = if py_truthy(data.get("error_description")) {
            py_str(data.get("error_description"))
        } else if let Some(code) = error_code.as_deref().filter(|c| !c.is_empty()) {
            code.to_string()
        } else {
            String::new()
        };
        if !description.is_empty() {
            detail = format!(": {description}");
        }
        if py_truthy(data.get("error_uri")) {
            detail.push_str(&format!(" (see {})", py_str(data.get("error_uri"))));
        }
    }
    TokenExchangeError::new(format!("{label}: HTTP {status}{detail}"), error_code, Some(status))
}

/// Which discovery error a metadata fetch reports once every URL failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataErrorKind {
    Metadata,
    ResourceMetadata,
}

// ------------------------------------------------------------------ URL helpers

/// The parts of Python's `urllib.parse.urlparse` this module uses.
#[derive(Debug, Clone, Default, PartialEq)]
struct ParsedUrl {
    scheme: String,
    netloc: String,
    path: String,
    query: String,
}

const SCHEME_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-.";

/// `urllib.parse.urlparse`: scheme (lowercased), netloc, path (without
/// `;params`), query; the fragment is dropped.
fn urlparse(url: &str) -> ParsedUrl {
    let mut rest = url;
    let mut scheme = String::new();
    if let Some(i) = rest.find(':') {
        let candidate = &rest[..i];
        if !candidate.is_empty()
            && candidate.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && candidate.chars().all(|c| SCHEME_CHARS.contains(c))
        {
            scheme = candidate.to_ascii_lowercase();
            rest = &rest[i + 1..];
        }
    }
    let mut netloc = String::new();
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        netloc = after[..end].to_string();
        rest = &after[end..];
    }
    // Fragment first, then query.
    if let Some(i) = rest.find('#') {
        rest = &rest[..i];
    }
    let (mut path, query) = match rest.find('?') {
        Some(i) => (rest[..i].to_string(), rest[i + 1..].to_string()),
        None => (rest.to_string(), String::new()),
    };
    // `;params` on the last path segment (schemes in `uses_params`).
    const USES_PARAMS: &[&str] =
        &["", "ftp", "hdl", "prospero", "http", "imap", "https", "shttp", "rtsp", "rtsps", "rtspu", "sip", "sips", "mms", "sftp", "tel"];
    if USES_PARAMS.contains(&scheme.as_str()) && path.contains(';') {
        let last_slash = path.rfind('/').map(|i| i + 1).unwrap_or(0);
        if let Some(semi) = path[last_slash..].find(';') {
            path.truncate(last_slash + semi);
        }
    }
    ParsedUrl { scheme, netloc, path, query }
}

/// `urllib.parse.quote_plus` with the default safe set.
fn quote_plus(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// `urllib.parse.urlencode` over ordered pairs.
fn urlencode(params: &[(&str, &str)]) -> String {
    params
        .iter()
        .map(|(k, v)| format!("{}={}", quote_plus(k), quote_plus(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// `secrets.token_urlsafe(32)`: 32 random bytes, base64url without padding.
fn token_urlsafe() -> String {
    let bytes: [u8; 32] = rand::random();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

fn string_list(data: &Map<String, Value>, key: &str) -> Vec<String> {
    match data.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .filter(|v| !v.is_null())
            .map(|v| py_str(Some(v)))
            .collect(),
        _ => Vec::new(),
    }
}

fn opt_i64(data: &Map<String, Value>, key: &str) -> Option<i64> {
    match data.get(key)? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse::<i64>().ok(),
        _ => None,
    }
}

// ------------------------------------------------------------------ service

/// Sync service for handling OAuth flows with remote MCP servers.
///
/// `new(None)` + `enter()` creates an owned client (30 s timeout, or the
/// process-wide test override — see
/// [`crate::mcp::client::set_http_transport_override`]); an injected client is
/// used as-is and never closed.
#[derive(Default)]
pub struct MCPOAuthService {
    http_client: Option<Arc<dyn HttpTransport>>,
    owns_client: bool,
}

impl fmt::Debug for MCPOAuthService {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MCPOAuthService")
            .field("open", &self.http_client.is_some())
            .field("owns_client", &self.owns_client)
            .finish()
    }
}

impl MCPOAuthService {
    pub const WELL_KNOWN_PATH: &'static str = "/.well-known/oauth-authorization-server";
    pub const WELL_KNOWN_OIDC_PATH: &'static str = "/.well-known/openid-configuration";
    pub const WELL_KNOWN_PRM_PATH: &'static str = "/.well-known/oauth-protected-resource";

    pub fn new(http_client: Option<reqwest::blocking::Client>) -> Self {
        let owns_client = http_client.is_none();
        Self {
            http_client: http_client.map(|c| Arc::new(ReqwestTransport::from_client(c)) as Arc<dyn HttpTransport>),
            owns_client,
        }
    }

    /// A service over an injected transport (`MCPOAuthService(http_client=fake)`).
    pub fn with_transport(transport: Arc<dyn HttpTransport>) -> Self {
        Self { http_client: Some(transport), owns_client: false }
    }

    /// Replace the transport in place (the tests' `svc._http_client = fake`).
    pub fn set_transport(&mut self, transport: Option<Arc<dyn HttpTransport>>) {
        self.http_client = transport;
    }

    /// Create the owned HTTP client (`__enter__`).
    pub fn enter(&mut self) -> &mut Self {
        if self.owns_client {
            self.http_client = Some(default_http_transport(30.0));
        }
        self
    }

    /// Close the owned HTTP client (`__exit__`).
    pub fn exit(&mut self) {
        if self.owns_client && self.http_client.is_some() {
            self.http_client = None;
        }
    }

    fn http(&self) -> Result<&Arc<dyn HttpTransport>, MCPOAuthError> {
        self.http_client
            .as_ref()
            .ok_or_else(|| MCPOAuthError::Other("'NoneType' object has no attribute 'get'".to_string()))
    }

    fn get_base_url(&self, server_url: &str) -> String {
        let parsed = urlparse(server_url);
        format!("{}://{}", parsed.scheme, parsed.netloc)
    }

    fn get_path_suffix(&self, server_url: &str) -> String {
        let parsed = urlparse(server_url);
        parsed.path.trim_start_matches('/').to_string()
    }

    /// Authorization-server metadata URLs to try, path-bearing forms first.
    pub fn get_metadata_urls_to_try(&self, server_url: &str) -> Vec<String> {
        let base_url = self.get_base_url(server_url);
        let path_suffix = self.get_path_suffix(server_url);

        let bare = format!("{base_url}{}", Self::WELL_KNOWN_PATH);
        if path_suffix.is_empty() {
            return vec![bare];
        }

        // For a path-scoped issuer, try the path-bearing well-known forms
        // first.  The bare (path-stripped) URL frequently describes a
        // *different* root authorization server that lacks a
        // registration_endpoint (e.g. Atlassian/Trello), so demote it to a
        // last resort rather than letting it shadow the tenant metadata.
        let issuer = format!("{base_url}/{path_suffix}");
        vec![
            format!("{base_url}{}/{path_suffix}", Self::WELL_KNOWN_PATH), // RFC 8414 well-known-inserted
            format!("{issuer}{}", Self::WELL_KNOWN_PATH),                 // OIDC-style path-append (oauth)
            format!("{issuer}{}", Self::WELL_KNOWN_OIDC_PATH),            // OIDC-style path-append (openid)
            bare,                                                          // last-resort fallback
        ]
    }

    /// Protected-resource metadata URLs to try.
    pub fn get_prm_urls_to_try(&self, server_url: &str) -> Vec<String> {
        let base_url = self.get_base_url(server_url);
        let path_suffix = self.get_path_suffix(server_url);

        let mut urls = vec![format!("{base_url}{}", Self::WELL_KNOWN_PRM_PATH)];
        if !path_suffix.is_empty() {
            urls.push(format!("{base_url}{}/{path_suffix}", Self::WELL_KNOWN_PRM_PATH));
        }
        urls
    }

    /// GET each URL in order, returning the first valid JSON object body.
    ///
    /// Falls through to the next candidate on ANY error (HTTP status such as
    /// 503, connection/transport errors, or invalid JSON) — not just 404 — so a
    /// transient failure on an earlier candidate never aborts discovery.
    /// Errors only once every candidate has been exhausted.
    fn fetch_first_metadata(
        &self,
        urls: &[String],
        kind: MetadataErrorKind,
        label: &str,
    ) -> Result<Map<String, Value>, MCPOAuthError> {
        let http = self.http()?;
        let mut last_error: Option<String> = None;
        for url in urls {
            match http.get(url) {
                Ok(response) if !response.is_success() => {
                    last_error = Some(format!("{url}: HTTP {}", response.status));
                }
                Ok(response) => match serde_json::from_str::<Value>(&response.body) {
                    Ok(Value::Object(data)) => return Ok(data),
                    Ok(_) => last_error = Some(format!("{url}: invalid JSON (expected an object)")),
                    Err(e) => last_error = Some(format!("{url}: invalid JSON ({e})")),
                },
                Err(e) => last_error = Some(format!("{url}: {e}")),
            }
        }

        let message = format!(
            "Failed to discover {label} from any endpoint. Last error: {}",
            last_error.as_deref().unwrap_or("None")
        );
        Err(match kind {
            MetadataErrorKind::Metadata => MetadataDiscoveryError(message).into(),
            MetadataErrorKind::ResourceMetadata => ResourceMetadataDiscoveryError(message).into(),
        })
    }

    /// Discover RFC 9728 metadata, honouring an advertised `metadata_url` first.
    pub fn discover_protected_resource_metadata(
        &self,
        server_url: &str,
        metadata_url: Option<&str>,
    ) -> Result<ProtectedResourceMetadata, MCPOAuthError> {
        // RFC 9728 §5.1: honor the metadata URL advertised in the 401
        // WWW-Authenticate header when present, then fall back to guessing the
        // well-known locations.
        let mut urls_to_try = self.get_prm_urls_to_try(server_url);
        if let Some(advertised) = metadata_url.filter(|u| !u.is_empty()) {
            if !urls_to_try.iter().any(|u| u == advertised) {
                urls_to_try.insert(0, advertised.to_string());
            }
        }

        let data = self.fetch_first_metadata(&urls_to_try, MetadataErrorKind::ResourceMetadata, "PRM")?;
        Ok(ProtectedResourceMetadata {
            resource: opt_string(&data, "resource"),
            authorization_servers: string_list(&data, "authorization_servers"),
            scopes_supported: string_list(&data, "scopes_supported"),
        })
    }

    /// Discover RFC 8414 / OIDC authorization server metadata.
    pub fn discover_oauth_metadata(&self, server_url: &str) -> Result<OAuthMetadata, MCPOAuthError> {
        let urls_to_try = self.get_metadata_urls_to_try(server_url);
        let data = self.fetch_first_metadata(&urls_to_try, MetadataErrorKind::Metadata, "OAuth metadata")?;
        Ok(OAuthMetadata {
            issuer: opt_string(&data, "issuer"),
            authorization_endpoint: opt_string(&data, "authorization_endpoint").unwrap_or_default(),
            token_endpoint: opt_string(&data, "token_endpoint").unwrap_or_default(),
            registration_endpoint: opt_string(&data, "registration_endpoint"),
            scopes_supported: string_list(&data, "scopes_supported"),
            response_types_supported: string_list(&data, "response_types_supported"),
            grant_types_supported: string_list(&data, "grant_types_supported"),
            code_challenge_methods_supported: string_list(&data, "code_challenge_methods_supported"),
            resource: opt_string(&data, "resource"),
            authorization_response_iss_parameter_supported: py_truthy(
                data.get("authorization_response_iss_parameter_supported"),
            ),
        })
    }

    /// RFC 7591 dynamic client registration (default `client_name` "Smartloop").
    pub fn register_client(
        &self,
        registration_endpoint: &str,
        redirect_uri: &str,
        client_name: &str,
    ) -> Result<ClientRegistrationResponse, MCPOAuthError> {
        let registration_data = serde_json::json!({
            "client_name": client_name,
            "redirect_uris": [redirect_uri],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "client_secret_post",
        });
        let headers = vec![("Content-Type".to_string(), "application/json".to_string())];

        let response = self
            .http()?
            .post(registration_endpoint, &HttpBody::Json(registration_data), &headers)
            .map_err(|e| ClientRegistrationError(format!("Failed to connect to {registration_endpoint}: {e}")))?;

        if !response.is_success() {
            let mut error_detail = String::new();
            if let Ok(Value::Object(error_data)) = serde_json::from_str::<Value>(&response.body) {
                let detail = if error_data.contains_key("error_description") {
                    py_str(error_data.get("error_description"))
                } else if error_data.contains_key("error") {
                    py_str(error_data.get("error"))
                } else {
                    String::new()
                };
                error_detail = format!(": {detail}");
            }
            return Err(ClientRegistrationError(format!(
                "Client registration failed: HTTP {}{error_detail}",
                response.status
            ))
            .into());
        }

        let data = match serde_json::from_str::<Value>(&response.body) {
            Ok(Value::Object(data)) => data,
            Ok(_) => return Err(MCPOAuthError::Other("Invalid registration response - not an object".to_string())),
            Err(e) => return Err(MCPOAuthError::Other(e.to_string())),
        };
        let Some(client_id) = data.get("client_id") else {
            return Err(ClientRegistrationError("Invalid registration response - missing 'client_id'".to_string()).into());
        };
        Ok(ClientRegistrationResponse {
            client_id: py_str(Some(client_id)),
            client_secret: opt_string(&data, "client_secret"),
            client_id_issued_at: opt_i64(&data, "client_id_issued_at"),
            client_secret_expires_at: opt_i64(&data, "client_secret_expires_at"),
        })
    }

    /// Canonical resource indicator (RFC 8707 §2) for an MCP server URL.
    ///
    /// Used when the server publishes no PRM `resource` of its own.  Lowercases
    /// scheme/host and drops the fragment, which RFC 8707 forbids; the path and
    /// query are significant (`/mcp` vs `/sse` are distinct resources) so they
    /// are preserved verbatim.
    pub fn canonical_resource(server_url: &str) -> String {
        let parsed = urlparse(server_url.trim());
        let path = if parsed.path == "/" { "" } else { parsed.path.as_str() };
        let canonical = format!("{}://{}{}", parsed.scheme.to_lowercase(), parsed.netloc.to_lowercase(), path);
        if parsed.query.is_empty() {
            canonical
        } else {
            format!("{canonical}?{}", parsed.query)
        }
    }

    /// `(code_verifier, code_challenge)` for PKCE S256.
    pub fn generate_pkce() -> (String, String) {
        let code_verifier = token_urlsafe();
        let digest = Sha256::digest(code_verifier.as_bytes());
        let code_challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest);
        (code_verifier, code_challenge)
    }

    /// A random URL-safe `state` value.
    pub fn generate_state() -> String {
        token_urlsafe()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_authorization_url(
        &self,
        authorization_endpoint: &str,
        client_id: &str,
        redirect_uri: &str,
        state: &str,
        code_challenge: &str,
        scope: Option<&str>,
        resource: Option<&str>,
    ) -> String {
        let mut params: Vec<(&str, &str)> = vec![
            ("response_type", "code"),
            ("client_id", client_id),
            ("redirect_uri", redirect_uri),
            ("state", state),
            ("code_challenge", code_challenge),
            ("code_challenge_method", "S256"),
        ];
        if let Some(scope) = scope.filter(|s| !s.is_empty()) {
            params.push(("scope", scope));
        }
        if let Some(resource) = resource.filter(|r| !r.is_empty()) {
            params.push(("resource", resource));
        }
        format!("{authorization_endpoint}?{}", urlencode(&params))
    }

    /// POST a token request and parse the RFC 6749 §5.1 response.
    fn token_request(
        &self,
        token_endpoint: &str,
        token_data: Vec<(String, String)>,
        label: &str,
    ) -> Result<TokenResponse, MCPOAuthError> {
        let headers = vec![("Content-Type".to_string(), "application/x-www-form-urlencoded".to_string())];
        let response = self
            .http()?
            .post(token_endpoint, &HttpBody::Form(token_data), &headers)
            .map_err(|e| TokenExchangeError::new(format!("Failed to connect to {token_endpoint}: {e}"), None, None))?;

        if !response.is_success() {
            return Err(token_error(response.status, &response.body, label).into());
        }

        let data = match serde_json::from_str::<Value>(&response.body) {
            Ok(Value::Object(data)) => data,
            Ok(_) => return Err(MCPOAuthError::Other("Invalid token response - not an object".to_string())),
            Err(e) => return Err(MCPOAuthError::Other(e.to_string())),
        };
        let Some(access_token) = data.get("access_token") else {
            return Err(TokenExchangeError::new("Invalid token response - missing 'access_token'", None, None).into());
        };
        Ok(TokenResponse {
            access_token: py_str(Some(access_token)),
            token_type: match data.get("token_type") {
                None => "Bearer".to_string(),
                Some(v) => py_str(Some(v)),
            },
            expires_in: opt_i64(&data, "expires_in"),
            refresh_token: opt_string(&data, "refresh_token"),
            scope: opt_string(&data, "scope"),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn exchange_code_for_tokens(
        &self,
        token_endpoint: &str,
        code: &str,
        redirect_uri: &str,
        client_id: &str,
        code_verifier: &str,
        client_secret: Option<&str>,
        resource: Option<&str>,
    ) -> Result<TokenResponse, MCPOAuthError> {
        let mut token_data = vec![
            ("grant_type".to_string(), "authorization_code".to_string()),
            ("code".to_string(), code.to_string()),
            ("redirect_uri".to_string(), redirect_uri.to_string()),
            ("client_id".to_string(), client_id.to_string()),
            ("code_verifier".to_string(), code_verifier.to_string()),
        ];
        if let Some(secret) = client_secret.filter(|s| !s.is_empty()) {
            token_data.push(("client_secret".to_string(), secret.to_string()));
        }
        if let Some(resource) = resource.filter(|r| !r.is_empty()) {
            token_data.push(("resource".to_string(), resource.to_string()));
        }
        self.token_request(token_endpoint, token_data, "Token exchange failed")
    }

    pub fn refresh_access_token(
        &self,
        token_endpoint: &str,
        refresh_token: &str,
        client_id: &str,
        client_secret: Option<&str>,
        resource: Option<&str>,
    ) -> Result<TokenResponse, MCPOAuthError> {
        let mut token_data = vec![
            ("grant_type".to_string(), "refresh_token".to_string()),
            ("refresh_token".to_string(), refresh_token.to_string()),
            ("client_id".to_string(), client_id.to_string()),
        ];
        if let Some(secret) = client_secret.filter(|s| !s.is_empty()) {
            token_data.push(("client_secret".to_string(), secret.to_string()));
        }
        if let Some(resource) = resource.filter(|r| !r.is_empty()) {
            token_data.push(("resource".to_string(), resource.to_string()));
        }
        self.token_request(token_endpoint, token_data, "Token refresh failed")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlparse_matches_python() {
        let p = urlparse("https://Auth.Example.com:443/a/b;p?x=1#frag");
        assert_eq!(p.scheme, "https");
        assert_eq!(p.netloc, "Auth.Example.com:443");
        assert_eq!(p.path, "/a/b");
        assert_eq!(p.query, "x=1");
        assert_eq!(urlparse("https://x").path, "");
    }

    #[test]
    fn quote_plus_matches_python() {
        assert_eq!(quote_plus("https://a.b/c d~*"), "https%3A%2F%2Fa.b%2Fc+d~%2A");
    }

    #[test]
    fn pkce_shapes() {
        let (v, c) = MCPOAuthService::generate_pkce();
        assert_eq!(v.len(), 43);
        assert_eq!(c.len(), 43);
        assert_ne!(MCPOAuthService::generate_state(), MCPOAuthService::generate_state());
    }
}
