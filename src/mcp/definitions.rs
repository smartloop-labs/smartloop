//! MCP connections a file describes, and how to launch them.
//!
//! A definition is the whole of what a `.toml` file says about one server: what
//! it is called, and how to reach it — a URL for a remote one, a command line for
//! a local one.  Dropping such a file into `~/.smartloop/mcp.d/` installs the
//! connection (see [`crate::mcp::definition_loader`] and
//! `POST /projects/{id}/mcp/definition`), which is what lets a server be added
//! without a code change or a trip through the UI.
//!
//! Nothing here indexes anything.  A connection's *tools* are enumerated once at
//! registration and that list is the catalog a step is resolved against (see
//! `crate::agents::mcp::mcp_agent::MCPAgent`); its *content* is read by calling
//! those tools when a question needs them.  This module used to also carry crawl
//! plans, and they are gone.  A file still carrying those fields installs anyway
//! — see [`RETIRED_FIELDS`] — because the connection it names is unaffected by
//! the indexing it can no longer ask for.
//!
//! Port of `smartloop/mcp/definitions.py`.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::workspace::MCPServer;

/// Re-exported: a definition's `command = "~/..."` is expanded against the
/// real (unsandboxed) home directory.
pub use crate::paths::{expand_home, real_home};

/// A definition that failed validation (pydantic `ValidationError`/`ValueError`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct MCPDefinitionError(pub String);

/// Lowercase `name` with separators stripped: `notion-search` → `notionsearch`.
///
/// The one place tool-name spelling is normalised.  Anything comparing tool or
/// argument names goes through it, so `notion-search`/`notion_search`/
/// `NotionSearch` are one name everywhere.
pub fn normalize_name(name: &str) -> String {
    todo!()
}

// ----------------------------------------------------------------- launchers

/// `"npx -y a-mcp"` → `(Some("npx"), ["-y", "a-mcp"])`.
///
/// Quoting is honoured (shell-style), so an argument with a space in it
/// survives as one argument.  An unclosed quote is an error.
pub fn split_command_line(text: &str) -> Result<(Option<String>, Vec<String>), MCPDefinitionError> {
    todo!()
}

/// `(command, args)` to run a bare package name.
///
/// A scoped npm name (`@scope/pkg`) runs under `npx -y`; anything else under
/// `uvx`.
pub fn launcher_for(package: &str) -> (String, Vec<String>) {
    todo!()
}

// --------------------------------------------------------------- retired keys

/// Per-entry fields the old indexing format used, accepted and ignored.
pub const RETIRED_FIELDS: &[&str] = &["crawl", "index", "index_every", "step", "extend"];

/// Which retired indexing fields `raw` declares, in the order it lists them.
pub fn retired_fields(raw: &Map<String, Value>) -> Vec<String> {
    todo!()
}

/// One MCP server, as a file describes it.
///
/// Two parts: **identity** (`name`, plus the `slug`/`match` that tie the file
/// to a server already installed) and **connection** (`url` for a remote
/// server, or `package`/`command` + `args` for one to run locally).  The
/// connection is optional.
///
/// Deserialization goes through [`TryFrom<Value>`]: retired fields are dropped,
/// a string `args` is shell-split, unknown keys are rejected and the
/// connection is validated.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(try_from = "serde_json::Value")]
pub struct MCPDefinition {
    pub name: String,
    /// A URL — makes the connection remote.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// A command line: a bare name runs under uvx/npx, anything else as written.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    /// The explicit form, used exactly as given.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: IndexMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Bearer token for remote servers that take one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Defaults to `name`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slug: Option<String>,
    /// Extra substrings matched against an installed server's name/package/URL.
    #[serde(rename = "match", default)]
    pub match_: Vec<String>,
}

impl TryFrom<Value> for MCPDefinition {
    type Error = MCPDefinitionError;

    /// Build a definition from a raw table, applying the model validators.
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        todo!()
    }
}

impl MCPDefinition {
    /// Build a definition from a raw table (`MCPDefinition(**raw)`).
    pub fn from_raw(raw: Map<String, Value>) -> Result<Self, MCPDefinitionError> {
        todo!()
    }

    /// A server is named one way, not two (`_check_connection`).
    pub fn check_connection(&self) -> Result<(), MCPDefinitionError> {
        todo!()
    }

    /// `"local"`, `"remote"`, or None when the file declares no connection.
    pub fn kind(&self) -> Option<&'static str> {
        todo!()
    }

    /// `(command, args)` for a local server.
    pub fn launch(&self) -> (Option<String>, Vec<String>) {
        todo!()
    }

    /// The slug this definition is registered and looked up under.
    pub fn key(&self) -> String {
        todo!()
    }

    /// Every substring that ties this definition to an installed server.
    pub fn match_terms(&self) -> Vec<String> {
        todo!()
    }
}

/// Lowercased name + package + url of an installed server, for matching.
fn server_haystack(server: &MCPServer) -> String {
    todo!()
}

/// The dropped-in definition that describes `server`, or None.
///
/// Matched on the catalog slug first, then on the substrings a definition
/// declares.
pub fn definition_for(server: &MCPServer) -> Option<MCPDefinition> {
    todo!()
}
