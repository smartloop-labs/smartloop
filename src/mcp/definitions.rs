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
//! plans — declarative walks that vectorized a server's pages, issues and cards
//! ahead of time — and they are gone: a walk went stale between questions, only
//! covered the listings someone had written a plan for, and answered from a
//! snapshot when the tool that would give the current answer was one call away.
//! A file still carrying those fields installs anyway — see [`RETIRED_FIELDS`] —
//! because the connection it names is unaffected by the indexing it can no
//! longer ask for.
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
/// `NotionSearch` are one name everywhere rather than in each caller's own
/// approximation of the rule.
pub fn normalize_name(name: &str) -> String {
    // Python's `str.lower()` lowercases Unicode too; only ASCII a-z0-9 survive.
    name.to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect()
}

// ----------------------------------------------------------------- launchers
//
// A local server is started by a command line, and which command that is
// belongs to the server: a Python one runs under `uvx`, a Node one under `npx`,
// and plenty of others are a module, a script or an absolute path.  Nothing
// here assumes one of them.

/// `shlex.split`: POSIX shell-style splitting; an unclosed quote is an error.
fn shlex_split(text: &str) -> Result<Vec<String>, MCPDefinitionError> {
    shlex::split(text).ok_or_else(|| MCPDefinitionError("No closing quotation".to_string()))
}

/// `"npx -y a-mcp"` → `(Some("npx"), ["-y", "a-mcp"])`.
///
/// Quoting is honoured (shell-style), so an argument with a space in it
/// survives as one argument.  An unclosed quote is a plain error naming the
/// field, which is better than the silent half-command a naive split would
/// produce.
pub fn split_command_line(text: &str) -> Result<(Option<String>, Vec<String>), MCPDefinitionError> {
    let mut parts = shlex_split(text.trim())?;
    if parts.is_empty() {
        return Ok((None, Vec::new()));
    }
    let command = parts.remove(0);
    Ok((Some(command), parts))
}

/// `(command, args)` to run a bare package name.
///
/// A scoped npm name (`@scope/pkg`) can only be an npm package, so it runs
/// under `npx -y`.  Anything else falls to `uvx`, which is what typing a bare
/// name into "Add your own" does — a file and that field have to install the
/// same server rather than two subtly different ones.  Every other launcher is
/// named explicitly.
pub fn launcher_for(package: &str) -> (String, Vec<String>) {
    if package.starts_with('@') {
        return ("npx".to_string(), vec!["-y".to_string(), package.to_string()]);
    }
    ("uvx".to_string(), vec![package.to_string()])
}

// ------------------------------------------------------------- home directory
//
// A definition writes a local server's path the way anyone would — `command =
// "~/.local/bin/craft"` — and `$HOME` is what expands the tilde.  Under the
// store sandboxes that env var is not the user's home, so the tilde is
// expanded against `crate::paths::real_home` instead; re-exported here because
// this is where `launch` reaches for it.

// --------------------------------------------------------------- retired keys
//
// Crawling is gone, but the files that configured it are still on people's
// disks, and a file that says `index_every = "15m"` beside a url or a package
// name still names a working connection.  So the retired keys are dropped
// rather than rejected: such a file installs, with a logged line naming the
// fields that did nothing.  Everything *else* unknown stays an error —
// `packag = ` has to name the typo rather than quietly install a server with
// no command.

/// Per-entry fields the old indexing format used, accepted and ignored.
pub const RETIRED_FIELDS: &[&str] = &["crawl", "index", "index_every", "step", "extend"];

/// Which retired indexing fields `raw` declares, in the order it lists them.
pub fn retired_fields(raw: &Map<String, Value>) -> Vec<String> {
    raw.keys()
        .filter(|k| RETIRED_FIELDS.contains(&k.as_str()))
        .cloned()
        .collect()
}

/// One MCP server, as a file describes it.
///
/// Two parts: **identity** (`name`, plus the `slug`/`match` that tie the file
/// to a server already installed) and **connection** (`url` for a remote
/// server, or `package`/`command` + `args` for one to run locally).  The
/// connection is optional — a file that only names a server still matches an
/// installed one, which is how a definition dropped in after the fact attaches
/// to a connection added through the UI.
///
/// Deserialization goes through [`TryFrom<Value>`]: retired fields are dropped,
/// a string `args` is shell-split, unknown keys are rejected and the
/// connection is validated.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
#[serde(try_from = "serde_json::Value")]
pub struct MCPDefinition {
    pub name: String,
    /// A URL or a package name — the same two things the "Add your own" field
    /// takes.  Which one is present makes a connection remote or local.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// A command line, not just a PyPI name: `package = "yahoo-finance-mcp"`
    /// runs under uvx, `package = "npx -y @foo/bar-mcp"` runs under npx, and
    /// `package = "python -m my.server"` runs as written.  See `launch`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<String>,
    /// The explicit form, used exactly as given — for a server whose launcher
    /// takes flags, or one that is a module or an absolute path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: IndexMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// For the remote servers that take a bearer token; OAuth is negotiated at
    /// install time and needs nothing here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// Defaults to `name`: a file usually describes one server and names it once.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slug: Option<String>,
    /// Extra substrings matched against an installed server's name, package
    /// and URL.  `name` and `slug` are always tried, so this is for aliases.
    #[serde(rename = "match", default)]
    pub match_: Vec<String>,
}

const KNOWN_FIELDS: &[&str] = &["name", "url", "package", "command", "args", "env", "cwd", "token", "slug", "match"];

/// Format collected `(location, message)` errors the way pydantic does.
fn validation_error(errors: &[(String, String)]) -> MCPDefinitionError {
    let mut out = format!(
        "{} validation error{} for MCPDefinition",
        errors.len(),
        if errors.len() == 1 { "" } else { "s" }
    );
    for (loc, msg) in errors {
        out.push('\n');
        if !loc.is_empty() {
            out.push_str(loc);
            out.push('\n');
        }
        out.push_str("  ");
        out.push_str(msg);
    }
    MCPDefinitionError(out)
}

fn input_repr(value: &Value) -> String {
    match value {
        Value::String(s) => format!("'{s}'"),
        other => crate::mcp::client::py_str(Some(other)),
    }
}

fn optional_string(
    raw: &Map<String, Value>,
    key: &str,
    errors: &mut Vec<(String, String)>,
) -> Option<String> {
    match raw.get(key) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => {
            errors.push((
                key.to_string(),
                format!("Input should be a valid string [type=string_type, input_value={}]", input_repr(other)),
            ));
            None
        }
    }
}

fn string_sequence(
    raw: &Map<String, Value>,
    key: &str,
    errors: &mut Vec<(String, String)>,
) -> Vec<String> {
    match raw.get(key) {
        None => Vec::new(),
        Some(Value::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(s) => out.push(s.clone()),
                    other => errors.push((
                        format!("{key}.{i}"),
                        format!("Input should be a valid string [type=string_type, input_value={}]", input_repr(other)),
                    )),
                }
            }
            out
        }
        Some(other) => {
            errors.push((
                key.to_string(),
                format!("Input should be a valid tuple [type=tuple_type, input_value={}]", input_repr(other)),
            ));
            Vec::new()
        }
    }
}

impl TryFrom<Value> for MCPDefinition {
    type Error = MCPDefinitionError;

    /// Build a definition from a raw table, applying the model validators.
    fn try_from(value: Value) -> Result<Self, Self::Error> {
        match value {
            Value::Object(raw) => Self::from_raw(raw),
            other => Err(validation_error(&[(
                String::new(),
                format!(
                    "Input should be a valid dictionary or instance of MCPDefinition [type=model_type, input_value={}]",
                    input_repr(&other)
                ),
            )])),
        }
    }
}

impl MCPDefinition {
    /// Build a definition from a raw table (`MCPDefinition(**raw)`).
    pub fn from_raw(raw: Map<String, Value>) -> Result<Self, MCPDefinitionError> {
        // Ignore the indexing fields a file written for the old format
        // carries.  Dropped in the model, so every way of building a
        // definition — a drop-in file, an upload, a test — treats them alike.
        let raw: Map<String, Value> = raw
            .into_iter()
            .filter(|(k, _)| !RETIRED_FIELDS.contains(&k.as_str()))
            .collect();

        let mut errors: Vec<(String, String)> = Vec::new();

        let name = match raw.get("name") {
            None => {
                errors.push(("name".to_string(), "Field required [type=missing]".to_string()));
                String::new()
            }
            Some(Value::String(s)) if s.chars().count() < 1 => {
                errors.push((
                    "name".to_string(),
                    "String should have at least 1 character [type=string_too_short, input_value='']".to_string(),
                ));
                String::new()
            }
            Some(Value::String(s)) => s.clone(),
            Some(other) => {
                errors.push((
                    "name".to_string(),
                    format!("Input should be a valid string [type=string_type, input_value={}]", input_repr(other)),
                ));
                String::new()
            }
        };
        let url = optional_string(&raw, "url", &mut errors);
        let package = optional_string(&raw, "package", &mut errors);
        let command = optional_string(&raw, "command", &mut errors);
        let cwd = optional_string(&raw, "cwd", &mut errors);
        let token = optional_string(&raw, "token", &mut errors);
        let slug = optional_string(&raw, "slug", &mut errors);

        // Accept `args = "-y some-mcp"` as well as a list: a command line is
        // how everyone writes these down, so a string is split the way a shell
        // would rather than rejected — or iterated one character per argument.
        let args = match raw.get("args") {
            Some(Value::String(s)) => match shlex_split(s.trim()) {
                Ok(parts) => parts,
                Err(e) => {
                    errors.push(("args".to_string(), format!("Value error, {e} [type=value_error]")));
                    Vec::new()
                }
            },
            _ => string_sequence(&raw, "args", &mut errors),
        };
        let match_ = string_sequence(&raw, "match", &mut errors);

        let mut env = IndexMap::new();
        match raw.get("env") {
            None => {}
            Some(Value::Object(map)) => {
                for (k, v) in map {
                    match v {
                        Value::String(s) => {
                            env.insert(k.clone(), s.clone());
                        }
                        other => errors.push((
                            format!("env.{k}"),
                            format!("Input should be a valid string [type=string_type, input_value={}]", input_repr(other)),
                        )),
                    }
                }
            }
            Some(other) => errors.push((
                "env".to_string(),
                format!("Input should be a valid dictionary [type=dict_type, input_value={}]", input_repr(other)),
            )),
        }

        for (key, value) in &raw {
            if !KNOWN_FIELDS.contains(&key.as_str()) {
                errors.push((
                    key.clone(),
                    format!(
                        "Extra inputs are not permitted [type=extra_forbidden, input_value={}]",
                        input_repr(value)
                    ),
                ));
            }
        }

        if !errors.is_empty() {
            return Err(validation_error(&errors));
        }

        let definition = Self { name, url, package, command, args, env, cwd, token, slug, match_ };
        definition.check_connection()?;
        Ok(definition)
    }

    /// A server is named one way, not two (`_check_connection`).
    pub fn check_connection(&self) -> Result<(), MCPDefinitionError> {
        let value_error = |msg: String| validation_error(&[(String::new(), format!("Value error, {msg} [type=value_error]"))]);
        let has = |v: &Option<String>| v.as_deref().is_some_and(|s| !s.is_empty());
        if !self.args.is_empty() && !(has(&self.command) || has(&self.package)) {
            return Err(value_error(format!(
                "'{}': args without a command — add `command = ` (the program to run) or use \
                 `package = ` for the whole command line",
                self.name
            )));
        }
        if has(&self.url) && (has(&self.package) || has(&self.command)) {
            return Err(value_error(format!(
                "'{}': a url and a package name are alternatives — a server is reached over HTTP \
                 or run as a process, not both",
                self.name
            )));
        }
        if has(&self.package) && has(&self.command) {
            return Err(value_error(format!(
                "'{}': package is shorthand for command + args, so setting both says the same thing twice",
                self.name
            )));
        }
        Ok(())
    }

    /// `"local"`, `"remote"`, or None when the file declares no connection.
    pub fn kind(&self) -> Option<&'static str> {
        let has = |v: &Option<String>| v.as_deref().is_some_and(|s| !s.is_empty());
        if has(&self.url) {
            return Some("remote");
        }
        if has(&self.package) || has(&self.command) {
            return Some("local");
        }
        None
    }

    /// `(command, args)` for a local server.
    ///
    /// `command` + `args` are taken as written; a `command` that carries its
    /// own arguments (`command = "npx -y a-mcp"`) is split, with any `args`
    /// appended.  `package` is the shorthand and it is a command line too: a
    /// bare name gets a launcher from [`launcher_for`], and anything with a
    /// space in it is already a command and is used as one.
    ///
    /// The launcher is never assumed beyond that single bare-name case.
    ///
    /// A command line with unbalanced quoting (which Python raises on) falls
    /// back to the whole string as the program; [`Self::try_launch`] reports it.
    pub fn launch(&self) -> (Option<String>, Vec<String>) {
        self.launch_with_home(crate::paths::real_home)
    }

    /// [`Self::launch`] with an injectable real-home resolver for `~`
    /// expansion (the tests' monkeypatched `smartloop.paths.real_home`).
    pub fn launch_with_home(&self, real_home: impl Fn() -> String + Copy) -> (Option<String>, Vec<String>) {
        match self.try_launch_with_home(real_home) {
            Ok(launch) => launch,
            Err(_) => match self.command.as_deref().filter(|c| !c.is_empty()) {
                Some(command) => (
                    Some(crate::paths::expand_home_with(command.trim(), real_home)),
                    self.args.clone(),
                ),
                None => (None, self.args.clone()),
            },
        }
    }

    /// [`Self::launch`], failing on a command line with unbalanced quoting.
    pub fn try_launch(&self) -> Result<(Option<String>, Vec<String>), MCPDefinitionError> {
        self.try_launch_with_home(crate::paths::real_home)
    }

    fn try_launch_with_home(
        &self,
        real_home: impl Fn() -> String + Copy,
    ) -> Result<(Option<String>, Vec<String>), MCPDefinitionError> {
        if let Some(command) = self.command.as_deref().filter(|c| !c.is_empty()) {
            let (command, embedded) = split_command_line(command)?;
            let mut args = embedded;
            args.extend(self.args.iter().cloned());
            return Ok((command.map(|c| crate::paths::expand_home_with(&c, real_home)), args));
        }
        if let Some(package) = self.package.as_deref().filter(|p| !p.is_empty()) {
            let (command, args) = split_command_line(package)?;
            let Some(command) = command else {
                return Ok((None, Vec::new()));
            };
            if !args.is_empty() {
                return Ok((Some(command), args));
            }
            let (command, args) = launcher_for(&command);
            return Ok((Some(command), args));
        }
        Ok((None, self.args.clone()))
    }

    /// The slug this definition is registered and looked up under.
    pub fn key(&self) -> String {
        let base = match self.slug.as_deref() {
            Some(slug) if !slug.is_empty() => slug,
            _ => self.name.as_str(),
        };
        base.trim().to_lowercase()
    }

    /// Every substring that ties this definition to an installed server.
    pub fn match_terms(&self) -> Vec<String> {
        let mut terms: Vec<String> = Vec::new();
        for term in [self.key(), self.name.to_lowercase()].into_iter().chain(self.match_.iter().cloned()) {
            if !terms.contains(&term) {
                terms.push(term);
            }
        }
        terms
    }
}

/// Lowercased name + package + url of an installed server, for matching.
fn server_haystack(server: &MCPServer) -> String {
    let mut parts: Vec<String> = vec![server.name.clone(), server.pkg.clone().unwrap_or_default()];
    if let Some(remote) = &server.remote_config {
        parts.push(remote.server_url.clone());
    }
    if let Some(config) = &server.config {
        parts.push(config.command.clone());
        parts.extend(config.args.iter().cloned());
    }
    parts.join(" ").to_lowercase()
}

/// The dropped-in definition that describes `server`, or None.
///
/// Matched on the catalog slug first — provenance the backend established
/// itself — then on the substrings a definition declares.  Used to find the
/// file to delete when a connection is removed, so a re-added server does not
/// silently inherit an old file.
pub fn definition_for(server: &MCPServer) -> Option<MCPDefinition> {
    let found = crate::mcp::definition_loader::definitions(None);
    if found.is_empty() {
        return None;
    }

    let slug = server.catalog_slug.as_deref().unwrap_or("").trim().to_lowercase();
    if !slug.is_empty() {
        if let Some(definition) = found.get(&slug) {
            return Some(definition.clone());
        }
    }

    let haystack = server_haystack(server);
    for definition in found.values() {
        if definition
            .match_terms()
            .iter()
            .any(|term| !term.is_empty() && haystack.contains(term.as_str()))
        {
            return Some(definition.clone());
        }
    }
    None
}
