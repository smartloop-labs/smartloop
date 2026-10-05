//! What to call *next*, once a server has answered.
//!
//! A listing tool answers "which records", not "what do they say".
//! `notion-search` comes back with page titles and ids; the document's text is
//! behind a second call, `notion-fetch(page_id=…)`.  That pairing is knowledge
//! about the server, and this is where it lives (`call_plans.toml` beside this
//! module, plus `[[plan]]` entries in any user file under `~/.smartloop/mcp.d/`).
//!
//! Given a payload from any listing tool, [`follow_up_calls`] returns the
//! per-record calls that fetch the bodies.
//!
//! Port of `smartloop/mcp/call_plans.py`.

use std::path::Path;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[allow(unused_imports)]
use crate::mcp::definition_loader::user_definition_paths;
#[allow(unused_imports)]
use crate::mcp::definitions::normalize_name;
use crate::workspace::MCPServer;

/// The bundled plans, embedded from `call_plans.toml` beside this module.
pub const BUNDLED_PLANS: &str = include_str!("call_plans.toml");

/// Label the bundled plans are logged under (Python: the file's path).
pub const BUNDLED_PLANS_PATH: &str = "call_plans.toml";

/// Argument names a per-record detail call takes the record's identifier under.
pub const ID_ARGS: &[&str] = &["id", "page_id", "pageId", "url", "uri", "identifier", "key", "gid"];

/// Keys a record's identifier is read from, in order of preference.
pub const RECORD_ID_KEYS: &[&str] = &[
    "id", "uuid", "gid", "key", "identifier", "page_id", "design_id", "shortLink", "url", "uri",
];

/// Keys whose value is the record list inside an envelope object.
pub const RECORD_LIST_KEYS: &[&str] = &[
    "results", "items", "records", "data", "nodes", "edges", "entries", "pages", "documents",
    "issues", "cards", "boards", "lists", "projects", "teams", "databases", "objects", "value",
    "content",
];

/// How many records one answer follows up on.
pub const DEFAULT_MAX_RECORDS: usize = 3;

fn default_detail_args() -> Vec<String> {
    ID_ARGS.iter().map(|s| s.to_string()).collect()
}

fn default_max_records() -> usize {
    DEFAULT_MAX_RECORDS
}

/// How to fetch the body of a record one of *this* server's listings named.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CallPlan {
    pub slug: String,
    /// Substrings matched (case-insensitively) against an installed server's
    /// name, package and URL.  `slug` is always tried.
    #[serde(rename = "match", default)]
    pub match_: Vec<String>,
    /// Candidate names for the tool that fetches one record, best first.
    #[serde(default)]
    pub detail_tools: Vec<String>,
    /// Argument names that tool takes the record's id under.
    #[serde(default = "default_detail_args")]
    pub detail_args: Vec<String>,
    /// Fixed arguments the detail call also needs, filtered against the tool's
    /// schema.
    #[serde(default)]
    pub detail_arguments: Map<String, Value>,
    #[serde(default = "default_max_records")]
    pub max_records: usize,
}

impl Default for CallPlan {
    fn default() -> Self {
        Self {
            slug: String::new(),
            match_: Vec::new(),
            detail_tools: Vec::new(),
            detail_args: default_detail_args(),
            detail_arguments: Map::new(),
            max_records: DEFAULT_MAX_RECORDS,
        }
    }
}

impl CallPlan {
    /// Field constraints (`slug` min length 1, `max_records >= 1`).
    pub fn validate(&self) -> anyhow::Result<()> {
        todo!()
    }

    /// Lowercased slug + match terms, de-duplicated in order.
    pub fn match_terms(&self) -> Vec<String> {
        todo!()
    }
}

// ------------------------------------------------------------------ loading

/// Parse `[[plan]]` entries out of one TOML document.
pub fn parse_plans(text: &str, source: &str) -> anyhow::Result<IndexMap<String, CallPlan>> {
    todo!()
}

/// The bundled plans, with any user file merged over them.
pub fn load_plans(home_dir: Option<&Path>) -> IndexMap<String, CallPlan> {
    todo!()
}

/// The plan registry, loaded once.
pub fn plans(home_dir: Option<&Path>) -> IndexMap<String, CallPlan> {
    todo!()
}

/// Drop the cache and read the plan files again.
pub fn reload_plans() -> IndexMap<String, CallPlan> {
    todo!()
}

// ------------------------------------------------------------ plan lookup

fn server_haystack(server: &MCPServer) -> String {
    todo!()
}

/// The follow-up plan for `server`, or None when it has no plan.
///
/// The catalog slug wins; the declared match terms are the fallback.
pub fn plan_for(server: &MCPServer) -> Option<CallPlan> {
    todo!()
}

// --------------------------------------------------------- schema helpers

/// Return a tool's input schema from `server`, or `{}`.
pub fn tool_schema(server: &MCPServer, tool_name: &str) -> Map<String, Value> {
    todo!()
}

fn properties(schema: &Map<String, Value>) -> Map<String, Value> {
    todo!()
}

/// The first of `candidates` the server actually exposes, or None.
///
/// Exact (normalized) equality, then prefix/suffix, then substring.
pub fn resolve_tool<S: AsRef<str>, C: AsRef<str>>(tool_names: &[S], candidates: &[C]) -> Option<String> {
    todo!()
}

/// The first of `candidates` the tool declares, or its lone required arg.
pub fn resolve_argument<C: AsRef<str>>(schema: &Map<String, Value>, candidates: &[C]) -> Option<String> {
    todo!()
}

/// Drop arguments the tool doesn't declare (an empty schema passes all).
pub fn filter_arguments(schema: &Map<String, Value>, arguments: &Map<String, Value>) -> Map<String, Value> {
    todo!()
}

/// Cast a record id to what the tool says the argument is.
pub fn coerce_argument(schema: &Map<String, Value>, name: &str, value: Value) -> Value {
    todo!()
}

// -------------------------------------------------------- record extraction

/// The first of `keys` whose value is a usable scalar.
fn first_str(data: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    todo!()
}

/// The record list inside whatever envelope a server wrapped it in.
fn unwrap_records(data: &Value) -> Vec<Value> {
    todo!()
}

/// Ids of the records `payload` lists, in order, at most `limit` of them.
pub fn record_ids(payload: &str, limit: usize) -> Vec<String> {
    todo!()
}

// ------------------------------------------------------------- the follow-up

/// `[(detail_tool, arguments), …]` to fetch the bodies `payload` names.
///
/// Empty when the connection has no plan, the detail tool is missing, the
/// call was already that detail tool, or the payload carries no ids.
pub fn follow_up_calls(server: &MCPServer, tool_name: &str, payload: &str) -> Vec<(String, Map<String, Value>)> {
    todo!()
}
