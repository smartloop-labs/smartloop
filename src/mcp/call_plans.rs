//! What to call *next*, once a server has answered.
//!
//! A listing tool answers "which records", not "what do they say".
//! `notion-search` comes back with page titles and ids; the pitch document's
//! text is behind a second call, `notion-fetch(page_id=…)`.  Nothing in a
//! tool's schema says the two go together — that pairing is knowledge about the
//! server, and this is where it lives:
//!
//! ```toml
//! [[plan]]
//! slug = "notion"
//! match = ["notion"]
//! detail_tools = ["notion-fetch", "API-get-block-children", "get_page"]
//! detail_args  = ["page_id", "pageId", "block_id", "id", "url"]
//! detail_arguments = { include_discussions = true }
//! ```
//!
//! Given a payload from any listing tool, [`follow_up_calls`] returns the
//! per-record calls that fetch the bodies — so an answer about a Notion page is
//! written from the page rather than from its title.
//!
//! This is the surviving half of the crawl plans: which tool fetches a record,
//! and which argument it takes the record's id under, applied to the call a
//! question just made rather than to a background pass.
//!
//! Plans are data (`call_plans.toml` beside this module, plus `[[plan]]`
//! entries in any user file under `~/.smartloop/mcp.d/`), so a server can be
//! taught without a code change.  A connection with no plan simply gets no
//! follow-up, which is right for the many servers whose listing *is* the
//! answer.
//!
//! Port of `smartloop/mcp/call_plans.py`.

use std::path::Path;
use std::sync::Arc;

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::mcp::definition_loader::{toml_to_json, user_definition_paths};
use crate::mcp::definitions::normalize_name;
use crate::workspace::{MCPServer, MCPToolDefinition};

/// Python logger name of this module.
const LOGGER: &str = "smartloop.mcp.call_plans";

/// The bundled plans, embedded from `call_plans.toml` beside this module.
pub const BUNDLED_PLANS: &str = include_str!("call_plans.toml");

/// Label the bundled plans are logged under (Python: the file's path).
pub const BUNDLED_PLANS_PATH: &str = "call_plans.toml";

/// Argument names a per-record detail call takes the record's identifier under.
pub const ID_ARGS: &[&str] = &["id", "page_id", "pageId", "url", "uri", "identifier", "key", "gid"];

/// Keys a record's identifier is read from, in order of preference.
///
/// Real identifiers first, bare spellings ahead of entity-qualified ones: a
/// server returning both means `id` by the record itself and `<entity>_id` by
/// something it points at, while the ones returning only `design_id` /
/// `page_id` have no other identity to offer.  `url` and `uri` come last, as
/// the fallback for listings that carry no id at all (hosted Notion is one).
pub const RECORD_ID_KEYS: &[&str] = &[
    "id", "uuid", "gid", "key", "identifier", "page_id", "design_id", "shortLink", "url", "uri",
];

/// Keys whose value is the record list inside an envelope object.
pub const RECORD_LIST_KEYS: &[&str] = &[
    "results", "items", "records", "data", "nodes", "edges", "entries", "pages", "documents",
    "issues", "cards", "boards", "lists", "projects", "teams", "databases", "objects", "value",
    "content",
];

/// How many records one answer follows up on.  A search returns twenty hits
/// and the question is about one or two of them; twenty round-trips to find
/// that out is a worse answer, later.
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
    /// Fixed arguments the detail call also needs — the content type to
    /// return, or the operation on a server where one tool covers several.
    /// Filtered against the tool's schema, so an argument it doesn't declare
    /// is dropped rather than rejecting the call.
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
    /// Build and validate a plan from a raw table (`CallPlan(**raw)`).
    pub fn from_raw(raw: Value) -> anyhow::Result<Self> {
        let plan: CallPlan = serde_json::from_value(raw)
            .map_err(|e| anyhow::anyhow!("1 validation error for CallPlan\n  {e}"))?;
        plan.validate()?;
        Ok(plan)
    }

    /// Field constraints (`slug` min length 1, `max_records >= 1`).
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.slug.chars().count() < 1 {
            anyhow::bail!(
                "1 validation error for CallPlan\nslug\n  String should have at least 1 character [type=string_too_short]"
            );
        }
        if self.max_records < 1 {
            anyhow::bail!(
                "1 validation error for CallPlan\nmax_records\n  Input should be greater than or equal to 1 [type=greater_than_equal]"
            );
        }
        Ok(())
    }

    /// Lowercased slug + match terms, de-duplicated in order.
    pub fn match_terms(&self) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        for term in std::iter::once(self.slug.to_lowercase()).chain(self.match_.iter().map(|m| m.to_lowercase())) {
            if !terms.contains(&term) {
                terms.push(term);
            }
        }
        terms
    }
}

/// The plan registry type: plans by lowercased slug.
pub type PlanRegistry = IndexMap<String, CallPlan>;

// ------------------------------------------------------------------ loading

static CACHE: Lazy<Mutex<Option<Arc<PlanRegistry>>>> = Lazy::new(|| Mutex::new(None));

/// Parse `[[plan]]` entries out of one TOML document.
pub fn parse_plans(text: &str, source: &str) -> anyhow::Result<PlanRegistry> {
    let document = toml_to_json(text)?;
    let mut found = PlanRegistry::new();
    let entries = match document.get("plan") {
        None | Some(Value::Null) => return Ok(found),
        Some(Value::Array(entries)) => entries.clone(),
        Some(Value::Object(o)) if o.is_empty() => return Ok(found),
        Some(other) => {
            // Python iterates whatever it is given: a table iterates its keys,
            // which are strings and so not tables.
            let _ = other;
            anyhow::bail!("each [[plan]] must be a table");
        }
    };
    for raw in entries {
        if !raw.is_object() {
            anyhow::bail!("each [[plan]] must be a table");
        }
        let plan = CallPlan::from_raw(raw)?;
        tracing::debug!(target: LOGGER, "[call-plans] {}: loaded '{}'", source, plan.slug);
        found.insert(plan.slug.to_lowercase(), plan);
    }
    Ok(found)
}

/// The bundled plans, with any user file merged over them.
pub fn load_plans(home_dir: Option<&Path>) -> PlanRegistry {
    let mut registry = PlanRegistry::new();
    match parse_plans(BUNDLED_PLANS, BUNDLED_PLANS_PATH) {
        Ok(found) => registry.extend(found),
        Err(e) => tracing::error!(target: LOGGER, "[call-plans] Skipping {}: {}", BUNDLED_PLANS_PATH, e),
    }
    for path in user_definition_paths(home_dir) {
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(target: LOGGER, "[call-plans] Skipping {}: {}", path.display(), e);
                continue;
            }
        };
        let parsed = std::str::from_utf8(&bytes)
            .map_err(anyhow::Error::from)
            .and_then(|text| parse_plans(text, &path.to_string_lossy()));
        match parsed {
            Ok(found) => registry.extend(found),
            // One bad file must not cost every other server its follow-up.
            Err(e) => tracing::warn!(target: LOGGER, "[call-plans] Skipping {}: {}", path.display(), e),
        }
    }
    registry
}

/// The plan registry, loaded once.
pub fn plans(home_dir: Option<&Path>) -> PlanRegistry {
    (*plans_shared(home_dir)).clone()
}

/// [`plans`] without the copy: the cached registry itself.
pub fn plans_shared(home_dir: Option<&Path>) -> Arc<PlanRegistry> {
    let mut cache = CACHE.lock();
    if cache.is_none() {
        let registry = Arc::new(load_plans(home_dir));
        let mut keys: Vec<&String> = registry.keys().collect();
        keys.sort();
        let names = keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", ");
        tracing::info!(
            target: LOGGER,
            "[call-plans] {} follow-up plan(s) loaded: {}",
            registry.len(),
            if names.is_empty() { "none" } else { names.as_str() }
        );
        *cache = Some(registry);
    }
    Arc::clone(cache.as_ref().expect("plans loaded"))
}

/// Drop the cache and read the plan files again.
pub fn reload_plans() -> PlanRegistry {
    (*reload_plans_shared()).clone()
}

/// [`reload_plans`], returning the freshly cached registry itself.
pub fn reload_plans_shared() -> Arc<PlanRegistry> {
    *CACHE.lock() = None;
    plans_shared(None)
}

// ------------------------------------------------------------ plan lookup

fn server_haystack(server: &MCPServer) -> String {
    let mut parts: Vec<String> = vec![
        server.name.clone(),
        server.pkg.clone().unwrap_or_default(),
        server.catalog_slug.clone().unwrap_or_default(),
    ];
    if let Some(remote) = &server.remote_config {
        parts.push(remote.server_url.clone());
    }
    if let Some(config) = &server.config {
        parts.push(config.command.clone());
        parts.extend(config.args.iter().cloned());
    }
    parts.join(" ").to_lowercase()
}

/// The follow-up plan for `server`, or None when it has no plan.
///
/// The catalog slug wins — it is provenance the backend established — and the
/// declared match terms are the fallback for a hand-added connection.
pub fn plan_for(server: &MCPServer) -> Option<CallPlan> {
    let registry = plans_shared(None);
    if registry.is_empty() {
        return None;
    }
    let slug = server.catalog_slug.as_deref().unwrap_or("").trim().to_lowercase();
    if !slug.is_empty() {
        if let Some(plan) = registry.get(&slug) {
            return Some(plan.clone());
        }
    }
    let haystack = server_haystack(server);
    registry
        .values()
        .find(|plan| {
            plan.match_terms()
                .iter()
                .any(|term| !term.is_empty() && haystack.contains(term.as_str()))
        })
        .cloned()
}

// --------------------------------------------------------- schema helpers

/// Return a tool's input schema from `server`, or `{}`.
pub fn tool_schema(server: &MCPServer, tool_name: &str) -> Map<String, Value> {
    tool_schema_in(&server.tools, tool_name)
}

/// [`tool_schema`] over a bare tool list.
pub fn tool_schema_in(tools: &[MCPToolDefinition], tool_name: &str) -> Map<String, Value> {
    tools
        .iter()
        .find(|tool| tool.name == tool_name)
        .map(|tool| tool.input_schema.clone().unwrap_or_default())
        .unwrap_or_default()
}

fn properties(schema: &Map<String, Value>) -> Map<String, Value> {
    match schema.get("properties") {
        Some(Value::Object(props)) => props.clone(),
        _ => Map::new(),
    }
}

/// The first of `candidates` the server actually exposes, or None.
///
/// Matching widens in three passes so the closest name always wins: exact
/// (normalized) equality, then a prefix/suffix match — which is what pairs a
/// plan's `fetch` with a server's `notion-fetch` — and finally a substring
/// match.  Candidate order decides ties, so a plan lists its preferred name
/// first.
pub fn resolve_tool<S: AsRef<str>, C: AsRef<str>>(tool_names: &[S], candidates: &[C]) -> Option<String> {
    if tool_names.is_empty() || candidates.is_empty() {
        return None;
    }
    let normalized: Vec<(&str, String)> = tool_names
        .iter()
        .map(|n| n.as_ref())
        .filter(|n| !n.is_empty())
        .map(|n| (n, normalize_name(n)))
        .collect();
    for candidate in candidates {
        let target = normalize_name(candidate.as_ref());
        if target.is_empty() {
            continue;
        }
        if let Some((name, _)) = normalized.iter().find(|(_, norm)| *norm == target) {
            return Some(name.to_string());
        }
        if let Some((name, _)) = normalized
            .iter()
            .find(|(_, norm)| norm.ends_with(target.as_str()) || norm.starts_with(target.as_str()))
        {
            return Some(name.to_string());
        }
        if let Some((name, _)) = normalized.iter().find(|(_, norm)| norm.contains(target.as_str())) {
            return Some(name.to_string());
        }
    }
    None
}

/// The first of `candidates` the tool declares, or its lone required arg.
///
/// A detail tool that takes exactly one required argument takes the record's
/// id under whatever it happens to call it.
pub fn resolve_argument<C: AsRef<str>>(schema: &Map<String, Value>, candidates: &[C]) -> Option<String> {
    let props = properties(schema);
    if !props.is_empty() {
        // Later keys win on a normalization collision, as in a dict literal.
        let mut by_norm: IndexMap<String, String> = IndexMap::new();
        for key in props.keys() {
            by_norm.insert(normalize_name(key), key.clone());
        }
        for candidate in candidates {
            if let Some(found) = by_norm.get(&normalize_name(candidate.as_ref())) {
                if !found.is_empty() {
                    return Some(found.clone());
                }
            }
        }
    }
    if let Some(Value::Array(required)) = schema.get("required") {
        if required.len() == 1 {
            if let Some(Value::String(only)) = required.first() {
                return Some(only.clone());
            }
        }
    }
    None
}

/// Drop arguments the tool doesn't declare.
///
/// A plan states what it *would* like to pass; a server that doesn't take it
/// would reject the whole call, so unknown keys go.  A tool with no declared
/// properties is passed them as given — an empty schema is telling us nothing,
/// not telling us "no arguments".
pub fn filter_arguments(schema: &Map<String, Value>, arguments: &Map<String, Value>) -> Map<String, Value> {
    if arguments.is_empty() {
        return Map::new();
    }
    let props = properties(schema);
    if props.is_empty() {
        return arguments.clone();
    }
    let mut by_norm: IndexMap<String, String> = IndexMap::new();
    for key in props.keys() {
        by_norm.insert(normalize_name(key), key.clone());
    }
    let mut resolved = Map::new();
    for (key, value) in arguments {
        if let Some(found) = by_norm.get(&normalize_name(key)) {
            if !found.is_empty() {
                resolved.insert(found.clone(), value.clone());
            }
        }
    }
    resolved
}

/// Python `int(str(value))` for a JSON scalar, else `None`.
fn py_int(text: &str) -> Option<Value> {
    let t = text.trim();
    let (sign, digits) = match t.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", t.strip_prefix('+').unwrap_or(t)),
    };
    // Digits with single underscores between them, as Python accepts.
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.chars().all(|c| c.is_ascii_digit() || c == '_')
    {
        return None;
    }
    let clean = format!("{sign}{}", digits.replace('_', ""));
    if let Ok(i) = clean.parse::<i64>() {
        return Some(Value::from(i));
    }
    if let Ok(u) = clean.parse::<u64>() {
        return Some(Value::from(u));
    }
    // Beyond 64 bits: keep the magnitude as a float rather than fail.
    clean.parse::<f64>().ok().and_then(serde_json::Number::from_f64).map(Value::Number)
}

/// Python `float(str(value))` for a JSON scalar, else `None`.
fn py_float(text: &str) -> Option<Value> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    let lower = t.to_ascii_lowercase();
    let body = lower.trim_start_matches(['+', '-']);
    let valid = matches!(body, "inf" | "infinity" | "nan")
        || (body.chars().all(|c| c.is_ascii_digit() || matches!(c, '.' | 'e' | '+' | '-' | '_'))
            && body.chars().any(|c| c.is_ascii_digit()));
    if !valid {
        return None;
    }
    let f: f64 = t.replace('_', "").parse().ok()?;
    // A non-finite float has no JSON form; the caller keeps the original.
    serde_json::Number::from_f64(f).map(Value::Number)
}

/// Cast a record id to what the tool says the argument is.
///
/// monday.com's board ids come back as strings and its item tools declare
/// numbers; sending the string is rejected by the server, and the id is the
/// one argument a follow-up cannot do without.
pub fn coerce_argument(schema: &Map<String, Value>, name: &str, value: Value) -> Value {
    let props = properties(schema);
    let Some(Value::Object(declared)) = props.get(name) else {
        return value;
    };
    let mut types: Vec<&str> = Vec::new();
    match declared.get("type") {
        Some(Value::String(t)) => types.push(t.as_str()),
        Some(Value::Array(items)) => types.extend(items.iter().filter_map(|t| t.as_str())),
        _ => {}
    }
    if types.contains(&"string") || types.is_empty() {
        return value;
    }
    if types.contains(&"number") || types.contains(&"integer") {
        let text = match &value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(true) => "True".to_string(),
            Value::Bool(false) => "False".to_string(),
            Value::Null => "None".to_string(),
            _ => return value,
        };
        if let Some(i) = py_int(&text) {
            return i;
        }
        if let Some(f) = py_float(&text) {
            return f;
        }
        return value;
    }
    value
}

// -------------------------------------------------------- record extraction

/// Python `str(value).strip()` for a usable scalar (str, int, float — and
/// bool, which Python counts as an int).
fn scalar_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(true) => Some("True".to_string()),
        Value::Bool(false) => Some("False".to_string()),
        _ => None,
    }
}

/// The first of `keys` whose value is a usable scalar.
fn first_str(data: &Map<String, Value>, keys: &[&str]) -> Option<String> {
    for key in keys {
        let Some(value) = data.get(*key) else { continue };
        if let Some(text) = scalar_text(value) {
            if !text.is_empty() {
                return Some(text);
            }
        }
        // Notion-style rich values: {"title": {"content": "..."}}
        if let Value::Object(nested) = value {
            if let Some(found) = first_str(nested, &["content", "plain_text", "text", "name", "title"]) {
                return Some(found);
            }
        }
    }
    None
}

/// The record list inside whatever envelope a server wrapped it in.
fn unwrap_records(data: &Value) -> Vec<Value> {
    match data {
        Value::Array(items) => items.clone(),
        Value::Object(obj) => {
            for key in RECORD_LIST_KEYS {
                match obj.get(*key) {
                    Some(Value::Array(items)) => return items.clone(),
                    Some(nested @ Value::Object(_)) => {
                        let found = unwrap_records(nested);
                        if !found.is_empty() {
                            return found;
                        }
                    }
                    _ => {}
                }
            }
            // A single record answered on its own is a list of one.
            if RECORD_ID_KEYS.iter().any(|k| obj.contains_key(*k)) {
                return vec![data.clone()];
            }
            Vec::new()
        }
        _ => Vec::new(),
    }
}

/// Ids of the records `payload` lists, in order, at most `limit` of them.
///
/// A tool result is text; most servers put JSON in it, and the ones that
/// don't have no ids to follow up on.  Both are ordinary — an unparseable
/// payload means no follow-up, not an error.
pub fn record_ids(payload: &str, limit: usize) -> Vec<String> {
    let text = payload.trim();
    if text.is_empty() {
        return Vec::new();
    }
    let start = [text.find('{'), text.find('[')].into_iter().flatten().min();
    let Some(start) = start else {
        return Vec::new();
    };
    let Ok(data) = serde_json::from_str::<Value>(&text[start..]) else {
        return Vec::new();
    };

    let mut ids: Vec<String> = Vec::new();
    for item in unwrap_records(&data) {
        // GraphQL edges
        let item = match &item {
            Value::Object(obj) if matches!(obj.get("node"), Some(Value::Object(_))) => obj["node"].clone(),
            _ => item,
        };
        let Value::Object(obj) = &item else { continue };
        let Some(record_id) = first_str(obj, RECORD_ID_KEYS) else { continue };
        if ids.contains(&record_id) {
            continue;
        }
        ids.push(record_id);
        if ids.len() >= limit {
            break;
        }
    }
    ids
}

// ------------------------------------------------------------- the follow-up

/// `[(detail_tool, arguments), …]` to fetch the bodies `payload` names.
///
/// Empty — the common case, and never an error — when the connection has no
/// plan, the plan's detail tool is not on this server, the call that produced
/// `payload` was already that detail tool, or the payload carries no ids.
pub fn follow_up_calls(server: &MCPServer, tool_name: &str, payload: &str) -> Vec<(String, Map<String, Value>)> {
    let Some(plan) = plan_for(server) else {
        return Vec::new();
    };
    if plan.detail_tools.is_empty() {
        return Vec::new();
    }

    let names: Vec<&str> = server.tools.iter().map(|t| t.name.as_str()).filter(|n| !n.is_empty()).collect();
    let Some(detail_tool) = resolve_tool(&names, &plan.detail_tools) else {
        return Vec::new();
    };
    if detail_tool == tool_name {
        return Vec::new();
    }

    let schema = tool_schema(server, &detail_tool);
    let Some(id_arg) = resolve_argument(&schema, &plan.detail_args) else {
        return Vec::new();
    };

    let fixed = filter_arguments(&schema, &plan.detail_arguments);
    record_ids(payload, plan.max_records)
        .into_iter()
        .map(|record_id| {
            let mut arguments = fixed.clone();
            arguments.insert(id_arg.clone(), coerce_argument(&schema, &id_arg, Value::String(record_id)));
            (detail_tool.clone(), arguments)
        })
        .collect()
}
