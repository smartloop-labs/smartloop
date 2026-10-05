//! Sync OAuth service for MCP server authentication (RFC 8414 / 9728 / 7591 /
//! 7636 / 8707 / 9207).
//!
//! Port of `smartloop/mcp/oauth.py`.  Blocking: call from a plain thread or
//! `tokio::task::spawn_blocking`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
    /// RFC 9207 §3: when set, an authorization response without `iss` must be
    /// rejected.
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
/// `error_code` is the OAuth 2.0 error identifier (RFC 6749 §5.2) when the
/// server sent one, and `None` for a transport failure.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct TokenExchangeError {
    pub message: String,
    pub error_code: Option<String>,
    pub status_code: Option<u16>,
}

impl TokenExchangeError {
    /// Error identifiers that mean the grant itself is finished.
    pub const DEAD_GRANT_CODES: &'static [&'static str] = &["invalid_grant", "invalid_client"];

    pub fn new(message: impl Into<String>, error_code: Option<String>, status_code: Option<u16>) -> Self {
        Self { message: message.into(), error_code, status_code }
    }

    /// True when re-minting the same refresh token can never work again.
    pub fn grant_is_dead(&self) -> bool {
        todo!()
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
    /// A bare `MCPOAuthError(message)`.
    #[error("{0}")]
    Other(String),
}

/// Build a [`TokenExchangeError`] from an RFC 6749 §5.2 error body.
///
/// `status` / `body` are the failed response's status and text; `label`
/// prefixes the message (`"{label}: HTTP {status}[: description][ (see uri)]"`).
pub fn token_error(status: u16, body: &str, label: &str) -> TokenExchangeError {
    todo!()
}

/// Which discovery error a metadata fetch reports once every URL failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetadataErrorKind {
    Metadata,
    ResourceMetadata,
}

/// Sync service for handling OAuth flows with remote MCP servers.
///
/// `new(None)` + `enter()` creates an owned client (30 s timeout); an injected
/// client is used as-is and never closed.
#[derive(Debug, Default)]
pub struct MCPOAuthService {
    http_client: Option<reqwest::blocking::Client>,
    owns_client: bool,
}

impl MCPOAuthService {
    pub const WELL_KNOWN_PATH: &'static str = "/.well-known/oauth-authorization-server";
    pub const WELL_KNOWN_OIDC_PATH: &'static str = "/.well-known/openid-configuration";
    pub const WELL_KNOWN_PRM_PATH: &'static str = "/.well-known/oauth-protected-resource";

    pub fn new(http_client: Option<reqwest::blocking::Client>) -> Self {
        let owns_client = http_client.is_none();
        Self { http_client, owns_client }
    }

    /// Create the owned HTTP client (`__enter__`).
    pub fn enter(&mut self) -> &mut Self {
        todo!()
    }

    /// Close the owned HTTP client (`__exit__`).
    pub fn exit(&mut self) {
        todo!()
    }

    fn get_base_url(&self, server_url: &str) -> String {
        todo!()
    }

    fn get_path_suffix(&self, server_url: &str) -> String {
        todo!()
    }

    /// Authorization-server metadata URLs to try, path-bearing forms first.
    pub fn get_metadata_urls_to_try(&self, server_url: &str) -> Vec<String> {
        todo!()
    }

    /// Protected-resource metadata URLs to try.
    pub fn get_prm_urls_to_try(&self, server_url: &str) -> Vec<String> {
        todo!()
    }

    /// GET each URL in order, returning the first valid JSON object body;
    /// falls through on any error.
    fn fetch_first_metadata(
        &self,
        urls: &[String],
        kind: MetadataErrorKind,
        label: &str,
    ) -> Result<Map<String, Value>, MCPOAuthError> {
        todo!()
    }

    /// Discover RFC 9728 metadata, honouring an advertised `metadata_url` first.
    pub fn discover_protected_resource_metadata(
        &self,
        server_url: &str,
        metadata_url: Option<&str>,
    ) -> Result<ProtectedResourceMetadata, MCPOAuthError> {
        todo!()
    }

    /// Discover RFC 8414 / OIDC authorization server metadata.
    pub fn discover_oauth_metadata(&self, server_url: &str) -> Result<OAuthMetadata, MCPOAuthError> {
        todo!()
    }

    /// RFC 7591 dynamic client registration (default `client_name` "Smartloop").
    pub fn register_client(
        &self,
        registration_endpoint: &str,
        redirect_uri: &str,
        client_name: &str,
    ) -> Result<ClientRegistrationResponse, MCPOAuthError> {
        todo!()
    }

    /// Canonical resource indicator (RFC 8707 §2) for an MCP server URL.
    pub fn canonical_resource(server_url: &str) -> String {
        todo!()
    }

    /// `(code_verifier, code_challenge)` for PKCE S256.
    pub fn generate_pkce() -> (String, String) {
        todo!()
    }

    /// A random URL-safe `state` value.
    pub fn generate_state() -> String {
        todo!()
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
        todo!()
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
        todo!()
    }

    pub fn refresh_access_token(
        &self,
        token_endpoint: &str,
        refresh_token: &str,
        client_id: &str,
        client_secret: Option<&str>,
        resource: Option<&str>,
    ) -> Result<TokenResponse, MCPOAuthError> {
        todo!()
    }
}
