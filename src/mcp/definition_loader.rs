//! Loading MCP connection definitions from TOML.
//!
//! Two sources, later merged over earlier:
//!
//! 1. `<home>/mcp.toml` — the user's own file, if they wrote one.
//! 2. `<home>/mcp.d/*.toml` — drop-ins, read in name order.
//!
//! A definition whose `slug` matches an earlier one replaces it.  A file that
//! fails to parse or validate is skipped with a logged reason and the rest still
//! load.
//!
//! Port of `smartloop/mcp/definition_loader.py`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use indexmap::IndexMap;

#[allow(unused_imports)]
use crate::config::AppSettings;
#[allow(unused_imports)]
use crate::mcp::definitions::{retired_fields, MCPDefinition};

/// Where a user's own definitions go, relative to the SLP home directory.
pub const USER_DEFINITIONS_FILE: &str = "mcp.toml";
pub const USER_DEFINITIONS_DIR: &str = "mcp.d";

/// Every user-supplied definition file, in the order they are merged.
///
/// `home_dir = None` resolves to `AppSettings::new().home_dir`.
pub fn user_definition_paths(home_dir: Option<&Path>) -> Vec<PathBuf> {
    todo!()
}

/// Parse one TOML document into `{key: MCPDefinition}`.
///
/// Errors on a malformed document; retired indexing fields are dropped and
/// logged once per entry that carries them.
pub fn parse_definitions(text: &str, source: &str) -> anyhow::Result<IndexMap<String, MCPDefinition>> {
    todo!()
}

/// Every definition the user's files declare, by key.
pub fn load_definitions(home_dir: Option<&Path>) -> IndexMap<String, MCPDefinition> {
    todo!()
}

/// (path, mtime) per definition file, so a dropped-in file is noticed.
fn stamp(home_dir: Option<&Path>) -> Vec<(PathBuf, SystemTime)> {
    todo!()
}

/// The definition registry, reloaded when any file has changed.
pub fn definitions(home_dir: Option<&Path>) -> IndexMap<String, MCPDefinition> {
    todo!()
}

/// Drop the cache and read every definition file again.
pub fn reload_definitions() -> IndexMap<String, MCPDefinition> {
    todo!()
}
