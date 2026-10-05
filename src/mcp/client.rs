//! Sync MCP client service for connecting to remote MCP servers and discovering
//! tools (JSON-RPC 2.0 over streamable HTTP, hand-rolled on blocking reqwest).
//!
//! Port of `smartloop/mcp/client.py`.  Blocking: call from a plain thread or
//! `tokio::task::spawn_blocking`.

use reqwest::header::HeaderMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::workspace::MCPToolDefinition;

// ------------------------------------------------------------------ errors

/// Connection-level failure (HTTP status, transport).  Carries the
/// `WWW-Authenticate` header and status on a 401/403.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{message}")]
pub struct MCPConnectionError {
    pub message: String,
    pub www_authenticate: Option<String>,
    pub status_code: Option<u16>,
}

impl MCPConnectionError {
    pub fn new(message: impl Into<String>, www_authenticate: Option<String>, status_code: Option<u16>) -> Self {
        Self { message: message.into(), www_authenticate, status_code }
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
    /// A bare `MCPClientError(message)`.
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
    /// The server rejected the token (HTTP 401), as on a tool call.
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
    #[serde(default)]
    pub auth_expired: bool,
}

// ------------------------------------------------------------------ client

/// Sync MCP client for connecting to remote MCP servers.
///
/// `enter()` opens the HTTP client (60 s timeout), `exit()` closes it and drops
/// the session (Python's context-manager protocol).
#[derive(Debug, Default)]
pub struct MCPClient {
    client: Option<reqwest::blocking::Client>,
    request_id: u64,
    session_id: Option<String>,
}

impl MCPClient {
    pub const PROTOCOL_VERSION: &'static str = "2024-11-05";
    pub const CLIENT_INFO_NAME: &'static str = "smartloop-cli";
    pub const CLIENT_INFO_VERSION: &'static str = "1.0.0";

    pub fn new() -> Self {
        Self::default()
    }

    /// `{"name": "smartloop-cli", "version": "1.0.0"}`.
    pub fn client_info() -> Value {
        todo!()
    }

    /// Open the HTTP client (`__enter__`).
    pub fn enter(&mut self) -> &mut Self {
        todo!()
    }

    /// Close the HTTP client and forget the session (`__exit__`).
    pub fn exit(&mut self) {
        todo!()
    }

    /// The current `Mcp-Session-Id`, if the server issued one.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    fn next_request_id(&mut self) -> String {
        todo!()
    }

    fn get_headers(&self, access_token: Option<&str>, token_type: &str) -> HeaderMap {
        todo!()
    }

    /// Parse the resource URI from a `WWW-Authenticate` header (RFC 6750 / 8707).
    pub fn parse_resource_from_www_authenticate(www_authenticate: &str) -> Option<String> {
        todo!()
    }

    /// Parse the PRM metadata URL from a `WWW-Authenticate` header (RFC 9728 §5.1).
    pub fn parse_resource_metadata_from_www_authenticate(www_authenticate: &str) -> Option<String> {
        todo!()
    }

    /// Extract the JSON-RPC message from an SSE response body.
    pub fn parse_sse_response(&self, response_text: &str) -> Result<Map<String, Value>, MCPClientError> {
        todo!()
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
        todo!()
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
        todo!()
    }

    /// Initialize, send `notifications/initialized`, then `tools/list`.
    pub fn connect_and_discover_tools(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> MCPConnectResult {
        todo!()
    }

    /// Perform the MCP initialize handshake if no session exists yet.
    fn ensure_session(
        &mut self,
        server_url: &str,
        access_token: Option<&str>,
        token_type: &str,
    ) -> Result<(), MCPClientError> {
        todo!()
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
        todo!()
    }
}

impl Drop for MCPClient {
    fn drop(&mut self) {
        self.client = None;
        self.session_id = None;
    }
}
