//! Loading MCP connection definitions from TOML.
//!
//! Two sources, later merged over earlier:
//!
//! 1. `<home>/mcp.toml` — the user's own file, if they wrote one.
//! 2. `<home>/mcp.d/*.toml` — drop-ins, read in name order.  This is where a
//!    connection goes: one file per server, added and removed without touching
//!    anything else.
//!
//! A definition whose `slug` matches an earlier one replaces it.  A file that
//! fails to parse or validate is skipped with a logged reason and the rest still
//! load: one malformed drop-in must not take the other connections down with it.
//!
//! `mcp.toml.example` beside this module documents the format with worked
//! examples; nothing reads it.
//!
//! Port of `smartloop/mcp/definition_loader.py`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

use indexmap::IndexMap;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, RwLock};
use serde_json::Value;

use crate::config::AppSettings;
use crate::mcp::definitions::{retired_fields, MCPDefinition};

/// Python logger name of this module.
const LOGGER: &str = "smartloop.mcp.definition_loader";

/// Where a user's own definitions go, relative to the SLP home directory.
pub const USER_DEFINITIONS_FILE: &str = "mcp.toml";
pub const USER_DEFINITIONS_DIR: &str = "mcp.d";

/// The registry type: definitions by key, in load order.
pub type DefinitionRegistry = IndexMap<String, MCPDefinition>;

/// Every user-supplied definition file, in the order they are merged.
///
/// `home_dir = None` resolves to `AppSettings::new().home_dir()`.
pub fn user_definition_paths(home_dir: Option<&Path>) -> Vec<PathBuf> {
    let home: PathBuf = match home_dir {
        Some(h) => h.to_path_buf(),
        None => PathBuf::from(AppSettings::new().home_dir()),
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    let single = home.join(USER_DEFINITIONS_FILE);
    if single.is_file() {
        paths.push(single);
    }
    // Escape the directory part so a home path containing glob
    // metacharacters still matches literally; `*` skips dot-files like
    // Python's glob.
    let pattern = format!(
        "{}/*.toml",
        glob::Pattern::escape(&home.join(USER_DEFINITIONS_DIR).to_string_lossy())
    );
    let options = glob::MatchOptions { require_literal_leading_dot: true, ..Default::default() };
    let mut dropins: Vec<PathBuf> = glob::glob_with(&pattern, options)
        .map(|it| it.filter_map(Result::ok).collect())
        .unwrap_or_default();
    dropins.sort_by(|a, b| a.to_string_lossy().cmp(&b.to_string_lossy()));
    paths.extend(dropins);
    paths
}

/// Parse a TOML document into a JSON value (tables become objects).
pub(crate) fn toml_to_json(text: &str) -> anyhow::Result<serde_json::Map<String, Value>> {
    let table: toml::Table = toml::from_str(text)?;
    match serde_json::to_value(table)? {
        Value::Object(map) => Ok(map),
        _ => anyhow::bail!("TOML document is not a table"),
    }
}

/// Python truthiness for the `document.get(key) or []` idiom.
fn is_empty_value(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(b) => !b,
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        Value::String(s) => s.is_empty(),
        Value::Number(n) => n.as_f64() == Some(0.0),
    }
}

/// Parse one TOML document into `{key: MCPDefinition}`.
///
/// Errors on a malformed document — callers decide whether one bad file is
/// fatal (it is, for an upload the user is watching) or not (it isn't, for the
/// drop-in directory).  The retired indexing fields are not malformed: they
/// are dropped, and logged once per entry that carries them.
pub fn parse_definitions(text: &str, source: &str) -> anyhow::Result<DefinitionRegistry> {
    let document = toml_to_json(text)?;
    let mut found = DefinitionRegistry::new();
    let entries = match document.get("mcp") {
        None => return Ok(found),
        Some(v) if is_empty_value(v) => return Ok(found),
        Some(Value::Array(entries)) => entries.clone(),
        Some(_) => anyhow::bail!("'mcp' must be a list of [[mcp]] tables"),
    };
    for raw in entries {
        let Value::Object(raw) = raw else {
            anyhow::bail!("each [[mcp]] must be a table");
        };
        let retired = retired_fields(&raw);
        let definition = MCPDefinition::from_raw(raw)?;
        let key = definition.key();
        if !retired.is_empty() {
            // The connection installed; the indexing it asked for no longer
            // exists.  Said out loud so the file can be tidied, rather than
            // leaving someone to wonder why nothing gets crawled.
            tracing::warn!(
                target: LOGGER,
                "[mcp-definitions] {}: '{}' declares {}, which no longer does anything — a \
                 connection's tools are called when a question needs them, nothing is indexed \
                 ahead of time",
                source, key, retired.join(", ")
            );
        }
        tracing::debug!(target: LOGGER, "[mcp-definitions] {}: loaded '{}'", source, key);
        found.insert(key, definition);
    }
    Ok(found)
}

/// Every definition the user's files declare, by key.
pub fn load_definitions(home_dir: Option<&Path>) -> DefinitionRegistry {
    let mut registry = DefinitionRegistry::new();
    for path in user_definition_paths(home_dir) {
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                tracing::warn!(target: LOGGER, "[mcp-definitions] Skipping {}: {}", path.display(), e);
                continue;
            }
        };
        let parsed = std::str::from_utf8(&bytes)
            .map_err(anyhow::Error::from)
            .and_then(|text| parse_definitions(text, &path.to_string_lossy()));
        match parsed {
            Ok(found) => registry.extend(found),
            // One bad drop-in must not take the working connections down.
            Err(e) => tracing::warn!(target: LOGGER, "[mcp-definitions] Skipping {}: {}", path.display(), e),
        }
    }
    registry
}

// --------------------------------------------------------------------- cache

type Stamp = Vec<(PathBuf, SystemTime)>;

struct Cache {
    registry: Option<Arc<DefinitionRegistry>>,
    stamp: Option<Stamp>,
}

static CACHE: Lazy<Mutex<Cache>> = Lazy::new(|| Mutex::new(Cache { registry: None, stamp: None }));

/// Replaces [`definitions`] / [`definitions_shared`] when set (the tests' monkeypatched
/// `smartloop.mcp.definition_loader.definitions`).
pub type DefinitionsOverride = Arc<dyn Fn() -> Arc<DefinitionRegistry> + Send + Sync>;

static OVERRIDE: Lazy<RwLock<Option<DefinitionsOverride>>> = Lazy::new(|| RwLock::new(None));

/// Install (or with `None`, remove) a replacement for [`definitions`].
/// Process-wide: tests using it must be serialized.
pub fn set_definitions_override(f: Option<DefinitionsOverride>) {
    *OVERRIDE.write() = f;
}

/// (path, mtime) per definition file, so a dropped-in file is noticed.
///
/// A handful of stat calls, which is what lets someone add a file and have the
/// next request see it rather than restarting the app.
fn stamp(home_dir: Option<&Path>) -> Stamp {
    user_definition_paths(home_dir)
        .into_iter()
        .filter_map(|path| {
            let mtime = std::fs::metadata(&path).and_then(|m| m.modified()).ok()?;
            Some((path, mtime))
        })
        .collect()
}

/// The definition registry, reloaded when any file has changed.
pub fn definitions(home_dir: Option<&Path>) -> DefinitionRegistry {
    (*definitions_shared(home_dir)).clone()
}

/// [`definitions`] without the copy: the cached registry itself, so callers
/// (and the tests' identity check) can tell a cache hit from a reload.
pub fn definitions_shared(home_dir: Option<&Path>) -> Arc<DefinitionRegistry> {
    if let Some(f) = OVERRIDE.read().clone() {
        return f();
    }
    let current = stamp(home_dir);
    let mut cache = CACHE.lock();
    let stale = match (&cache.registry, &cache.stamp) {
        (Some(_), Some(s)) => *s != current,
        _ => true,
    };
    if stale {
        let registry = Arc::new(load_definitions(home_dir));
        let mut keys: Vec<&String> = registry.keys().collect();
        keys.sort();
        let names = keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", ");
        tracing::info!(
            target: LOGGER,
            "[mcp-definitions] {} definition(s) loaded: {}",
            registry.len(),
            if names.is_empty() { "none" } else { names.as_str() }
        );
        cache.registry = Some(registry);
        cache.stamp = Some(current);
    }
    Arc::clone(cache.registry.as_ref().expect("registry loaded"))
}

/// Drop the cache and read every definition file again.
pub fn reload_definitions() -> DefinitionRegistry {
    (*reload_definitions_shared()).clone()
}

/// [`reload_definitions`], returning the freshly cached registry itself.
pub fn reload_definitions_shared() -> Arc<DefinitionRegistry> {
    {
        let mut cache = CACHE.lock();
        cache.registry = None;
        cache.stamp = None;
    }
    definitions_shared(None)
}
