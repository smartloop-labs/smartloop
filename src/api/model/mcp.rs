//! Port of `smartloop/api/model/mcp.py`.
//!
//! Request/response models for MCP server API endpoints.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

/// A tool exposed by an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPToolResponse {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Map<String, Value>>,
}

fn default_local() -> String {
    "local".to_string()
}
fn default_remote() -> String {
    "remote".to_string()
}
fn default_ready() -> String {
    "ready".to_string()
}
fn default_true() -> bool {
    true
}
fn default_empty_args() -> Option<Vec<String>> {
    Some(Vec::new())
}

/// A registered MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MCPServerResponse {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default = "default_local")]
    pub server_type: String,
    #[serde(default)]
    pub server_url: Option<String>,
    /// Slug of the curated catalog entry this server came from, when it did.
    #[serde(default)]
    pub catalog_slug: Option<String>,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub system: bool,
    #[serde(default)]
    pub tools: Vec<MCPToolResponse>,
    #[serde(default)]
    pub server_info: Option<Map<String, Value>>,
    /// Background provisioning state: "pending" while discovery, enrichment and
    /// indexing run on the MCP task, then "ready" or "failed" with a reason.
    /// Registration returns immediately, so clients watch this rather than
    /// waiting on the response.
    #[serde(default = "default_ready")]
    pub status: String,
    #[serde(default)]
    pub last_error: Option<String>,
    /// Account this connection is linked as, when the server can say.
    #[serde(default)]
    pub account: Option<String>,
}

impl Default for MCPServerResponse {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: String::new(),
            description: None,
            tags: Vec::new(),
            command: None,
            args: Vec::new(),
            cwd: None,
            server_type: default_local(),
            server_url: None,
            catalog_slug: None,
            enabled: true,
            system: false,
            tools: Vec::new(),
            server_info: None,
            status: default_ready(),
            last_error: None,
            account: None,
        }
    }
}

/// List of MCP servers.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPServerListResponse {
    pub servers: Vec<MCPServerResponse>,
}

/// Response after deleting an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct DeleteMCPServerResponse {
    pub id: String,
    pub deleted: bool,
    pub message: String,
}

/// Request to register an MCP server (local stdio or remote HTTP).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MCPRegisterRequest {
    /// "local" or "remote".
    #[serde(default = "default_remote")]
    pub server_type: String,
    /// Display name (optional for local, derived from args).
    #[serde(default)]
    pub name: Option<String>,
    /// Stdio command for local server, e.g. "uvx", "npx".
    #[serde(default)]
    pub command: Option<String>,
    /// Command args for local server.
    #[serde(default = "default_empty_args")]
    pub args: Option<Vec<String>>,
    // Local server fields
    #[serde(default)]
    pub env: HashMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    // Remote server fields
    #[serde(default)]
    pub server_url: Option<String>,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub callback_url: Option<String>,
    /// Offered as a link on the OAuth success page (no automatic navigation).
    #[serde(default)]
    pub return_url: Option<String>,
    /// Install a curated catalog entry: the backend resolves the slug and takes
    /// server_url, description and tags from the catalog rather than the caller,
    /// so curated metadata cannot drift or be spoofed. `server_url` above is
    /// kept as the fallback for when resolution misses.
    #[serde(default)]
    pub catalog_slug: Option<String>,
}

impl Default for MCPRegisterRequest {
    fn default() -> Self {
        Self {
            server_type: default_remote(),
            name: None,
            command: None,
            args: default_empty_args(),
            env: HashMap::new(),
            cwd: None,
            server_url: None,
            token: None,
            callback_url: None,
            return_url: None,
            catalog_slug: None,
        }
    }
}

/// A connection described by a dropped-in TOML file.
///
/// `toml` is the file's text; `filename` names the copy kept under
/// `~/.smartloop/mcp.d/`, so the connection survives as a file.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPDefinitionRequest {
    pub toml: String,
    #[serde(default)]
    pub filename: Option<String>,
}

/// Unified response for MCP server registration.
///
/// If `auth_type` is 'oauth', `auth_url` and oauth session fields are returned.
/// Otherwise, the server is registered directly and server fields are populated.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPRegisterResponse {
    /// 'none', 'token', 'oauth'.
    pub auth_type: String,
    /// Populated when server is registered directly (no OAuth).
    #[serde(default)]
    pub server: Option<MCPServerResponse>,
    // Populated when OAuth is required
    #[serde(default)]
    pub auth_url: Option<String>,
    #[serde(default)]
    pub state: Option<String>,
    #[serde(default)]
    pub code_verifier: Option<String>,
    #[serde(default)]
    pub client_id: Option<String>,
    #[serde(default)]
    pub client_secret: Option<String>,
    #[serde(default)]
    pub token_endpoint: Option<String>,
    #[serde(default)]
    pub authorization_endpoint: Option<String>,
    /// RFC 8707 resource indicator the client must echo back on the token request.
    #[serde(default)]
    pub resource: Option<String>,
    #[serde(default)]
    pub server_url: Option<String>,
}

/// OAuth callback payload completing a remote MCP registration.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPOAuthCallbackRequest {
    pub server_url: String,
    pub code: String,
    pub state: String,
    pub code_verifier: String,
    pub client_id: String,
    #[serde(default)]
    pub client_secret: Option<String>,
    pub token_endpoint: String,
    pub authorization_endpoint: String,
    #[serde(default)]
    pub resource: Option<String>,
    #[serde(default)]
    pub callback_url: Option<String>,
}

/// Request to call a tool on an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPToolCallRequest {
    pub tool_name: String,
    /// Values are str/int/float/bool/list/dict.
    #[serde(default)]
    pub arguments: Option<Map<String, Value>>,
}

/// Result of an MCP tool call.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MCPToolCallResponse {
    pub success: bool,
    #[serde(default)]
    pub content: Option<Vec<Map<String, Value>>>,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default)]
    pub message: Option<String>,
}
