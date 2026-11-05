//! Port of `smartloop/agents/mcp/mcp_agent.py`.
//!
//! MCPAgent — resolves an MCP step to a concrete tool call, arguments included.
//!
//! Registering a server enumerates its tools once (see
//! `ModelService::provision_mcp_server`) and persists what it found: the name,
//! the description and the JSON input schema of every tool.  That list *is* the
//! catalog, and it is already in memory on every request — so resolution reads
//! it directly.  The tools are shown to the model as callable signatures and it
//! answers with the one call to make, its required arguments filled from the
//! request and from what earlier steps returned.
//!
//! This replaces a vector index over tool descriptions.  Embedding a one-line
//! description and taking the top-K by cosine distance answered "which tool
//! reads like this sentence", which is not the question being asked: a step
//! names an *operation*, and the tool that serves it is the one whose arguments
//! the context can actually fill.  Retrieval also had to run before argument
//! extraction, so the two halves of a single decision were made by two calls
//! that could not see each other — the tool was picked without knowing whether
//! its arguments could be satisfied, and a wrong pick was then argued for
//! confidently.  One call over the whole catalog costs no embedding pass at
//! registration, no per-project vector collection, and no re-index when a
//! server's tools change.
//!
//! A model that can reason *always* reasons here.  The call passes
//! `thinking=Some(true)` and an output budget above the floor under which
//! `LlamaCpp::thinking_for_budget` suppresses the block, so neither the user's
//! thinking setting nor a small default budget can turn it off.  The setting is
//! about the visible answer; this call's output is a JSON tool call whose
//! monologue is stripped before anything reads it (`crate::llms::thinking`),
//! and choosing a tool while working out whether its arguments can be filled is
//! precisely the judgement the block exists for.  The flag travels with the
//! request, so it holds whether the weights are in this process or behind the
//! model host.
//!
//! Which model runs it is decided one level up: `pipeline_builder` routes the
//! resolve node to a model declaring `ModelCapability::Thinking` when the
//! workspace has one enabled, rather than to whichever model the step would
//! otherwise use.
//!
//! ## Native tool calling
//!
//! A model that calls functions natively (`BaseLLM::supports_native_tools`:
//! the catalog declares `ModelCapability::Tools` — a large model such as a
//! Qwen 27B — and its chat template or API takes a `tools` list) is handed the
//! catalog as real tool definitions, full JSON schemas included, and answers
//! in its own trained call format ([`crate::llms::tool_calls`]).  It sees the
//! optional parameters too, and may fill them when the request states their
//! value — the description, labels and priority of a ticket it is asked to
//! create — where the numbered prompt shows required parameters only.
//!
//! This is the same resolve node in the same DAG: only *how* the call is
//! worked out differs.  The answer is a [`ResolvedCall`] like any other, and
//! the call, the result mapping and everything downstream are unchanged.  A
//! native attempt that yields no usable call (the catalog does not fit, the
//! reply does not parse, the model names no listed tool) falls back to the
//! prompt-based resolution below, on the same model.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::planner::{mcp_connection_is_usable, FETCH_SERVER_ID, SYSTEM_TOOLS_SERVER_ID};
use crate::constants::display_model_label;
use crate::llms::tool_calls::{parse_tool_calls, with_tools};
use crate::llms::BaseLLM;
use crate::workspace::workspace::MCPServer;

/// Logger name of the Python module.
pub const LOGGER: &str = "smartloop.agents.mcp.mcp_agent";

/// Output budget for the resolve call when the model reasons.  Above
/// `LlamaCpp::THINKING_MIN_BUDGET` (512) so the reasoning block is not
/// suppressed: the JSON answer is ~60 tokens and the rest is room to think.
pub const RESOLVE_MAX_TOKENS: usize = 1024;

/// Output budget for a native tool call.  Larger than [`RESOLVE_MAX_TOKENS`]:
/// a native call may carry optional arguments with real prose in them (a
/// ticket's description), on top of the reasoning block.
pub const NATIVE_RESOLVE_MAX_TOKENS: usize = 2048;

/// Longest function name an OpenAI-compatible endpoint accepts.
const MAX_FUNCTION_NAME: usize = 64;

// Catalog trimming.  The whole tool list goes into one prompt now, so it is
// bounded here rather than by retrieval: a long description is guidance the
// model needs the gist of, not all of.
pub const MAX_TOOL_DESC: usize = 200;
pub const MAX_PARAM_DESC: usize = 140;
pub const MAX_CATALOG_CHARS: usize = 12000;
/// Never render less than this, whatever the window says — below it the list
/// is too abbreviated to pick from at all.
pub const MIN_CATALOG_CHARS: usize = 4000;
pub const MAX_CONTEXT_CHARS: usize = 2000;

/// How many arguments of a resolved call are shown to the user (see [`format_call`]).
pub const MAX_SHOWN_ARGS: usize = 4;
pub const MAX_SHOWN_ARG_CHARS: usize = 60;

/// How many resolutions are remembered process-wide.
///
/// One question resolves the same way twice in a single request: once on the
/// collect path (Explorer runs a read-only tool while gathering context) and
/// again when the plan reaches its TOOL step.  The second call would ask the
/// same model the same question over the same catalog — seconds of reasoning
/// for an answer already in hand — so the answer is kept.  Keyed by everything
/// that could change it, tiny, and process-wide because the two callers hold
/// different `MCPAgent` instances.
pub const CACHE_SIZE: usize = 4;

/// Cache key: (normalized directive, hash of the trimmed context, qualified tool names).
pub type CacheKey = (String, u64, Vec<String>);

/// The process-wide resolution cache (LRU, oldest first).
static RESOLUTION_CACHE: Lazy<Mutex<VecDeque<(CacheKey, Option<ResolvedCall>)>>> =
    Lazy::new(|| Mutex::new(VecDeque::new()));

/// Forget every remembered resolution (tests, and a workspace reload).
pub fn clear_resolution_cache() {
    RESOLUTION_CACHE.lock().clear();
}

/// Build the cache key for one question.
fn cache_key(directive: &str, context: Option<&str>, entries: &[Entry]) -> CacheKey {
    let normalized = directive.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let mut hasher = DefaultHasher::new();
    context.unwrap_or("").trim().hash(&mut hasher);
    (
        normalized,
        hasher.finish(),
        entries.iter().map(|e| e.qualified.clone()).collect(),
    )
}

/// One tool call: where it lives, what it is, and what to pass it.
///
/// `arguments` holds the tool's *required* parameters, as far as the request
/// and the context could fill them — the caller decides whether a partial fill
/// is worth calling with (see `StepExecutor::resolve_tool_arguments`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResolvedCall {
    pub server_id: String,
    pub tool_name: String,
    pub arguments: Map<String, Value>,
    pub reasoned: bool,
    /// Resolved through the model's native tool calling, so `arguments` may
    /// hold optional parameters the request stated, not only required ones.
    pub native: bool,
    /// The further calls the model asked for in the same answer, in order —
    /// a native model plans "find the user, then file the issue" as two
    /// calls at once.  This step makes only the first; the collect pass
    /// hands a write among these to the planner (see `Explorer`).
    pub also: Vec<ResolvedCall>,
}

impl ResolvedCall {
    /// A call with no arguments that was not reasoned over.
    pub fn new(server_id: impl Into<String>, tool_name: impl Into<String>) -> Self {
        Self {
            server_id: server_id.into(),
            tool_name: tool_name.into(),
            arguments: Map::new(),
            reasoned: false,
            native: false,
            also: Vec::new(),
        }
    }
}

/// Tools this account cannot call, learned by calling them.  Keyed
/// `(server_id, tool_name)` and held for the life of the process: an
/// entitlement is a property of the workspace's plan, not of one request, and
/// re-learning it costs a wasted resolution and a wasted round-trip every turn.
static UNAVAILABLE_TOOLS: Lazy<Mutex<HashSet<(String, String)>>> =
    Lazy::new(|| Mutex::new(HashSet::new()));

/// Phrasings a server uses to decline a call it advertised, in the shape of a
/// successful result.  Deliberately narrow: matching "upgrade" or "plan" alone
/// would throw away a real Notion page about pricing, which is exactly the
/// content a user would ask for.  Each of these is a sentence about the
/// *caller's* entitlement, not something a document says.
pub static ENTITLEMENT_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    [
        r"(?i)\brequires? an?\b[^.]{0,40}\bplan\b",
        r"(?i)\brequires? a subscription\b",
        r"(?i)\bnot available on your (current )?plan\b",
        r"(?i)\bmcp_tool_upsell\b",
        r"(?i)\bupgrade (your plan|to a paid)\b",
    ]
    .iter()
    .map(|p| Regex::new(p).expect("valid entitlement pattern"))
    .collect()
});

/// An entitlement notice is short — a sentence and a link.  A long payload that
/// happens to contain one of the phrases above is far more likely to be real
/// content that mentions plans, so length is part of the test.
pub const MAX_ENTITLEMENT_CHARS: usize = 2000;

/// Whether `text` is a server declining a call rather than answering it.
///
/// A paywalled tool does not fail.  Notion's `notion-ai-search` on a
/// non-Business plan returns HTTP 200, `success=True`, and a body reading
/// "This tool requires a Business plan or higher" with an upsell link — which
/// every layer above then treats as retrieved content, so an advertisement
/// reaches the model in place of the document and the answer is written from
/// it.  There is no field distinguishing this from an answer; the text is the
/// only signal there is.
pub fn is_entitlement_notice(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let candidate = text.trim();
    if candidate.chars().count() > MAX_ENTITLEMENT_CHARS {
        return false;
    }
    ENTITLEMENT_PATTERNS.iter().any(|p| p.is_match(candidate))
}

/// Stop offering `tool_name` for resolution on this server.
///
/// Called when a tool declines for entitlement reasons.  Without this the
/// resolver keeps picking the same paywalled tool every turn — it is still in
/// discovery, still the best-sounding name for the question — and every turn
/// pays a resolution and a round-trip to be told the same thing.
pub fn mark_tool_unavailable(server_id: &str, tool_name: &str) {
    if server_id.is_empty() || tool_name.is_empty() {
        return;
    }
    UNAVAILABLE_TOOLS
        .lock()
        .insert((server_id.to_string(), tool_name.to_string()));
}

/// The `(server_id, tool_name)` pairs known to be unavailable.
pub fn unavailable_tools() -> HashSet<(String, String)> {
    UNAVAILABLE_TOOLS.lock().clone()
}

/// Clear what has been learned — for tests, and for a re-authentication.
pub fn forget_unavailable_tools() {
    UNAVAILABLE_TOOLS.lock().clear();
}

// ------------------------------------------------------------ Python repr

/// Python's `repr()` of a string: single quotes unless the text contains a
/// single quote and no double quote, with backslashes, the chosen quote and
/// non-printable characters escaped.
pub fn py_repr_str(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if c.is_control() || (c.is_whitespace() && c != ' ') => {
                let code = c as u32;
                if code < 0x100 {
                    out.push_str(&format!("\\x{code:02x}"));
                } else if code < 0x10000 {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    out.push_str(&format!("\\U{code:08x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python's `repr()` of a float (`1.0`, `0.0001`, `1e-05`, `1e+16`, `inf`).
fn py_repr_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if !(-4..16).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{mantissa}e{sign}{:02}", exp.abs());
    }
    let plain = format!("{f}");
    if plain.contains('.') {
        plain
    } else {
        format!("{plain}.0")
    }
}

/// Python's `repr()` of a JSON value as the Python object it decodes to
/// (`None`, `True`, `'text'`, `[1, 2]`, `{'k': 'v'}`).
pub fn py_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(u) = n.as_u64() {
                u.to_string()
            } else {
                py_repr_float(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => py_repr_str(s),
        Value::Array(items) => {
            let parts: Vec<String> = items.iter().map(py_repr).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}: {}", py_repr_str(k), py_repr(v)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// Python's `str()` of a JSON value: strings as-is, everything else as `repr`.
pub fn py_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => py_repr(other),
    }
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The first `n` characters of `s`.
fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `notion.create-pages(parent='Roadmap', pages=[…])` — the call, for a person.
///
/// What was decided is the interesting part of an MCP step, and until this
/// existed the client only saw which *model* had been chosen: a tool ran, or
/// silently did not, with the call itself visible in the server logs alone.
/// Values are truncated because an argument can be a whole document, and
/// rendered with Python `repr` so an empty string or a stray newline is visible
/// instead of vanishing into the punctuation.
pub fn format_call(server_name: &str, tool_name: &str, arguments: &Map<String, Value>) -> String {
    let qualified = if server_name.is_empty() {
        tool_name.to_string()
    } else {
        format!("{server_name}.{tool_name}")
    };
    if arguments.is_empty() {
        return format!("{qualified}()");
    }
    let mut parts: Vec<String> = Vec::new();
    for (key, value) in arguments.iter().take(MAX_SHOWN_ARGS) {
        let is_str = value.is_string();
        let mut shown = match value {
            Value::String(s) => s.clone(),
            other => py_repr(other),
        };
        if shown.chars().count() > MAX_SHOWN_ARG_CHARS {
            shown = take_chars(&shown, MAX_SHOWN_ARG_CHARS) + "…";
        }
        if is_str {
            parts.push(format!("{key}={}", py_repr_str(&shown)));
        } else {
            parts.push(format!("{key}={shown}"));
        }
    }
    if arguments.len() > MAX_SHOWN_ARGS {
        parts.push(format!("+{} more", arguments.len() - MAX_SHOWN_ARGS));
    }
    format!("{qualified}({})", parts.join(", "))
}

/// True for the values Python's `value is None or value == "" or value == []
/// or value == {}` treats as unfilled.
fn is_unfilled(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => true,
        Some(Value::String(s)) => s.is_empty(),
        Some(Value::Array(a)) => a.is_empty(),
        Some(Value::Object(o)) => o.is_empty(),
        _ => false,
    }
}

/// Required parameters of `input_schema` that `arguments` leaves unfilled.
pub fn missing_required(input_schema: &Map<String, Value>, arguments: &Map<String, Value>) -> Vec<String> {
    let mut missing = Vec::new();
    if let Some(Value::Array(required)) = input_schema.get("required") {
        for name in required {
            let value = match name {
                Value::String(key) => arguments.get(key),
                _ => None,
            };
            if is_unfilled(value) {
                missing.push(py_str(name));
            }
        }
    }
    missing
}

/// One catalog entry (Python `MCPAgent._Entry`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Entry {
    pub server_id: String,
    pub server_name: String,
    pub tool_name: String,
    pub qualified: String,
    pub description: String,
    pub schema: Map<String, Value>,
}

/// The `required` names of a schema that are also declared in its properties.
fn declared_required(schema: &Map<String, Value>, properties: &Map<String, Value>) -> Vec<String> {
    match schema.get("required") {
        Some(Value::Array(required)) => required
            .iter()
            .filter_map(Value::as_str)
            .filter(|r| properties.contains_key(*r))
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

/// `schema.get("properties") or {}` as an object.
fn schema_properties(schema: &Map<String, Value>) -> Map<String, Value> {
    match schema.get("properties") {
        Some(Value::Object(p)) => p.clone(),
        _ => Map::new(),
    }
}

/// `info.get("type", "any")` rendered the way an f-string would.
fn type_label(info: &Value) -> String {
    match info.get("type") {
        Some(t) => py_str(t),
        None => "any".to_string(),
    }
}

/// Resolves a step to a tool call by reasoning over the discovered tools.
pub struct MCPAgent {
    llm: Arc<dyn BaseLLM>,
    /// The argument extractor, used for its value checks only (built lazily).
    validators: OnceLock<Option<MCPArgumentExtractor>>,
}

impl MCPAgent {
    /// Create an agent that resolves with `llm` unless a call overrides it.
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm, validators: OnceLock::new() }
    }

    // ------------------------------------------------------------------ resolve

    /// Return the tool call `directive` asks for, or `None`.
    ///
    /// `context` is what earlier steps produced, in whatever shape the caller
    /// already had it — the values a chained call needs ("close the issue I
    /// just filed") live there and nowhere else, so resolution reads it
    /// alongside the request.
    ///
    /// `llm` overrides the model.  Callers pass the step's capability model:
    /// this is the harder of the judgements an MCP step makes, and leaving it
    /// on the small router model while upgrading only the arguments got the
    /// arguments right for the wrong tool.
    pub fn resolve(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<Arc<dyn BaseLLM>>,
    ) -> Option<ResolvedCall> {
        self.resolve_with(directive, mcp_servers, context, llm.as_deref())
    }

    /// [`MCPAgent::resolve`] with a borrowed override model (what
    /// `ToolResolver::resolve` receives).
    pub fn resolve_with(
        &self,
        directive: &str,
        mcp_servers: &[MCPServer],
        context: Option<&str>,
        llm: Option<&dyn BaseLLM>,
    ) -> Option<ResolvedCall> {
        if directive.is_empty() {
            return None;
        }

        let entries = self.catalog(mcp_servers);
        if entries.is_empty() {
            tracing::info!(target: LOGGER, "[mcp-agent] no discovered tools to resolve against");
            return None;
        }

        // A lone tool that takes no required arguments is already resolved;
        // there is nothing for the model to decide or fill.
        if entries.len() == 1 && !entries[0].schema.get("required").map(truthy).unwrap_or(false) {
            let only = &entries[0];
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] single argument-free tool → {}",
                only.qualified
            );
            return Some(ResolvedCall::new(only.server_id.clone(), only.tool_name.clone()));
        }

        let key = cache_key(directive, context, &entries);
        {
            let mut cache = RESOLUTION_CACHE.lock();
            if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
                let item = cache.remove(pos).expect("position is in range");
                let cached = item.1.clone();
                cache.push_back(item);
                tracing::info!(
                    target: LOGGER,
                    "[mcp-agent] reusing this question's resolution: {}",
                    match &cached {
                        Some(c) => format_call("", &c.tool_name, &c.arguments),
                        None => "no tool".to_string(),
                    }
                );
                return cached;
            }
        }

        let effective_llm: &dyn BaseLLM = match llm {
            Some(l) => l,
            None => self.llm.as_ref(),
        };

        if effective_llm.supports_native_tools() {
            if let Some(resolved) = self.resolve_native(directive, context, &entries, effective_llm) {
                Self::remember(key, Some(resolved.clone()));
                return Some(resolved);
            }
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] native tool call gave no usable call — falling back to prompt-based resolution"
            );
        }

        let prompt = self.prompt(directive, context, &entries, Some(effective_llm));
        let reasoned = Self::can_think(effective_llm);
        let budget = Self::budget(effective_llm, &prompt, RESOLVE_MAX_TOKENS);

        tracing::info!(
            target: LOGGER,
            "[mcp-agent] resolving over {} discovered tool(s) with {} (thinking={}, max_tokens={})",
            entries.len(),
            Self::model_label(effective_llm),
            if reasoned { "True" } else { "False" },
            budget
        );

        // thinking=true regardless of what `reasoned` says: a model that
        // cannot reason ignores it, and one that can must not have this call
        // decided by the user's answer-formatting setting.  `invoke` never
        // fails — an error comes back as text, which parses to no tool.
        let text = effective_llm.invoke(&prompt, Some(budget), Some(true));

        let (choice, raw_arguments) = self.parse(&text, &entries);
        let Some(choice) = choice else {
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] no tool resolved from: {}",
                take_chars(text.trim(), 200)
            );
            Self::remember(key, None);
            return None;
        };

        let entry = &entries[choice];
        let arguments = self.clean_arguments(&entry.schema, &raw_arguments, false);
        tracing::info!(
            target: LOGGER,
            "[mcp-agent] resolved {}",
            format_call(&entry.server_name, &entry.tool_name, &arguments)
        );
        let resolved = ResolvedCall {
            server_id: entry.server_id.clone(),
            tool_name: entry.tool_name.clone(),
            arguments,
            reasoned,
            native: false,
            also: Vec::new(),
        };
        Self::remember(key, Some(resolved.clone()));
        Some(resolved)
    }

    // ------------------------------------------------------------------ native

    /// Resolve with the model's own tool calling, or `None` to fall back.
    pub fn resolve_native(
        &self,
        directive: &str,
        context: Option<&str>,
        entries: &[Entry],
        llm: &dyn BaseLLM,
    ) -> Option<ResolvedCall> {
        let (tools, names) = self.native_tools(entries);
        let size: usize = tools.iter().map(|t| t.to_string().chars().count()).sum();
        let budget = self.catalog_budget(llm);
        if size > budget {
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] {} tool definition(s) need {} chars, over the {} this model can be shown — not calling natively",
                tools.len(),
                size,
                budget
            );
            return None;
        }

        let prompt = with_tools(&self.native_prompt(directive, context), &tools);
        let reasoned = Self::can_think(llm);
        let max_tokens = Self::budget(llm, &prompt, NATIVE_RESOLVE_MAX_TOKENS);
        tracing::info!(
            target: LOGGER,
            "[mcp-agent] resolving natively over {} tool(s) with {} (thinking={}, max_tokens={})",
            tools.len(),
            Self::model_label(llm),
            if reasoned { "True" } else { "False" },
            max_tokens
        );
        // Reasoning forced on for the same reason as the prompt path.
        let text = llm.invoke(&prompt, Some(max_tokens), Some(true));

        let calls = parse_tool_calls(&text);
        let found = calls.iter().find_map(|call| Self::native_entry(&call.name, &names, entries).map(|i| (i, call)));
        let Some((index, call)) = found else {
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] no listed tool in the native reply ({} call(s)): {}",
                calls.len(),
                take_chars(text.trim(), 200)
            );
            return None;
        };
        let also: Vec<ResolvedCall> = calls
            .iter()
            .filter(|other| !std::ptr::eq(*other, call))
            .filter_map(|other| {
                let i = Self::native_entry(&other.name, &names, entries)?;
                let e = &entries[i];
                Some(ResolvedCall {
                    server_id: e.server_id.clone(),
                    tool_name: e.tool_name.clone(),
                    arguments: self.clean_arguments(&e.schema, &other.arguments, true),
                    reasoned,
                    native: true,
                    also: Vec::new(),
                })
            })
            .collect();
        if !also.is_empty() {
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] model asked for {} calls — this step makes the first; then {}",
                calls.len(),
                also.iter().map(|c| c.tool_name.as_str()).collect::<Vec<_>>().join(", ")
            );
        }

        let entry = &entries[index];
        let arguments = self.clean_arguments(&entry.schema, &call.arguments, true);
        tracing::info!(
            target: LOGGER,
            "[mcp-agent] resolved natively {}",
            format_call(&entry.server_name, &entry.tool_name, &arguments)
        );
        Some(ResolvedCall {
            server_id: entry.server_id.clone(),
            tool_name: entry.tool_name.clone(),
            arguments,
            reasoned,
            native: true,
            also,
        })
    }

    /// OpenAI function definitions for `entries`, and the function name each
    /// entry was given (parallel to `entries`).
    ///
    /// A tool keeps its own name when no other server lists the same one, so
    /// the model sees the name it was trained on; a clash is qualified with
    /// the server.  Names are cut to what an endpoint accepts.  Descriptions
    /// are trimmed as in the prompt catalog; the schema is passed whole.
    pub fn native_tools(&self, entries: &[Entry]) -> (Vec<Value>, Vec<String>) {
        let sanitize = |raw: &str| -> String {
            let name: String = raw
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
                .collect();
            take_chars(&name, MAX_FUNCTION_NAME)
        };
        let mut names: Vec<String> = Vec::with_capacity(entries.len());
        for entry in entries {
            let clashes = entries.iter().filter(|e| e.tool_name == entry.tool_name).count() > 1;
            let base = if clashes {
                sanitize(&format!("{}__{}", entry.server_name, entry.tool_name))
            } else {
                sanitize(&entry.tool_name)
            };
            let mut name = base.clone();
            let mut n = 2;
            while names.contains(&name) {
                let suffix = format!("_{n}");
                name = format!("{}{suffix}", take_chars(&base, MAX_FUNCTION_NAME - suffix.len()));
                n += 1;
            }
            names.push(name);
        }

        let tools = entries
            .iter()
            .zip(&names)
            .map(|(entry, name)| {
                let mut description = format!("[{}]", entry.server_name);
                if !entry.description.is_empty() {
                    description.push(' ');
                    description.push_str(&Self::trim(&entry.description, MAX_TOOL_DESC * 2));
                }
                json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": Self::native_parameters(&entry.schema),
                    }
                })
            })
            .collect();
        (tools, names)
    }

    /// A tool's input schema as a function's `parameters`: an object schema
    /// always, with each property's description trimmed.
    fn native_parameters(schema: &Map<String, Value>) -> Value {
        let mut parameters = schema.clone();
        parameters.insert("type".into(), json!("object"));
        let properties = schema_properties(schema)
            .into_iter()
            .map(|(name, mut info)| {
                if let Some(desc) = info.get("description").and_then(Value::as_str) {
                    let trimmed = Self::trim(desc, MAX_PARAM_DESC);
                    info["description"] = Value::String(trimmed);
                }
                (name, info)
            })
            .collect();
        parameters.insert("properties".into(), Value::Object(properties));
        Value::Object(parameters)
    }

    /// The entry a native call names: by the name it was given, else by the
    /// tool's own or qualified name (a model may echo either).
    pub fn native_entry(name: &str, names: &[String], entries: &[Entry]) -> Option<usize> {
        let name = name.trim();
        names
            .iter()
            .position(|n| n == name)
            .or_else(|| entries.iter().position(|e| e.qualified == name))
            .or_else(|| {
                let mut matching = entries.iter().enumerate().filter(|(_, e)| e.tool_name == name);
                match (matching.next(), matching.next()) {
                    (Some((i, _)), None) => Some(i),
                    _ => None,
                }
            })
    }

    /// The messages of a native resolution; the tools travel beside them.
    pub fn native_prompt(&self, directive: &str, context: Option<&str>) -> String {
        let mut context_block = String::new();
        if let Some(ctx) = context.filter(|c| !c.trim().is_empty()) {
            context_block = format!(
                "\n\nContext from earlier steps (use it to fill arguments):\n{}",
                Self::trim(ctx, MAX_CONTEXT_CHARS)
            );
        }
        format!(
            "<system>
You carry out one step of a plan by calling exactly ONE of the provided tools.

- The request names what the user wants, not the tool. A request about content
  a connected server holds is served by that server's search or read tool;
  reading is cheap and a near miss is useful.
- Fill the tool's arguments from the request and the context. Include an
  optional argument only when its value is stated or clearly implied there —
  for something you are asked to create (a ticket, an issue, a page), that
  means its title, description, labels, priority and the like.
- Never invent an id, a cursor, a token or a URL.
- If no tool could serve the request, answer with the single word NONE.
</system>
<user>
Request: {directive}{context_block}
</user>"
        )
    }

    /// Keep this resolution for the other caller in the same request.
    fn remember(key: CacheKey, resolved: Option<ResolvedCall>) {
        let mut cache = RESOLUTION_CACHE.lock();
        if let Some(pos) = cache.iter().position(|(k, _)| *k == key) {
            cache.remove(pos);
        }
        cache.push_back((key, resolved));
        while cache.len() > CACHE_SIZE {
            cache.pop_front();
        }
    }

    // ------------------------------------------------------------------ catalog

    /// Every callable external tool, as discovery recorded it.
    ///
    /// The system server is left out: its tools (bash, grep, file-read/write,
    /// explore) are resolved by the planner against their own constants, and
    /// putting them in this list only gives the model a way to answer "read a
    /// file" to a question about Notion.
    ///
    /// Tools discovery advertised but the account cannot actually call are
    /// left out too — see [`mark_tool_unavailable`].  Discovery lists what a
    /// server *has*, not what this plan is entitled to, so a paywalled tool
    /// looks identical to a usable one until it is called.
    pub fn catalog(&self, mcp_servers: &[MCPServer]) -> Vec<Entry> {
        let unavailable = unavailable_tools();
        let mut entries: Vec<Entry> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for server in mcp_servers {
            let server_id = server.id.as_str();
            if server_id.is_empty() || server_id == SYSTEM_TOOLS_SERVER_ID || server_id == FETCH_SERVER_ID {
                continue;
            }
            if server.system {
                continue;
            }
            if !mcp_connection_is_usable(server) {
                continue;
            }
            let server_name = if server.name.is_empty() { server_id } else { server.name.as_str() };
            for tool in &server.tools {
                let name = tool.name.as_str();
                if name.is_empty() {
                    continue;
                }
                if unavailable.contains(&(server_id.to_string(), name.to_string())) {
                    continue;
                }
                let qualified = format!("{server_name}.{name}");
                if seen.contains(&qualified) {
                    continue;
                }
                seen.insert(qualified.clone());
                entries.push(Entry {
                    server_id: server_id.to_string(),
                    server_name: server_name.to_string(),
                    tool_name: name.to_string(),
                    qualified,
                    description: tool.description.as_deref().unwrap_or("").trim().to_string(),
                    schema: tool.input_schema.clone().unwrap_or_default(),
                });
            }
        }
        entries
    }

    /// One catalog entry: the call, its purpose, and the arguments it needs.
    ///
    /// Only *required* parameters are listed.  An optional one the model can
    /// see is one it can fabricate — a priority filter or a stale cursor that
    /// silently narrows a listing to nothing — and the tool applies its own
    /// defaults for everything left unset.
    ///
    /// `compact` drops the prose and keeps the call shape, for a catalog too
    /// long to render in full (see [`MCPAgent::prompt`]).
    pub fn signature(&self, entry: &Entry, number: usize, compact: bool) -> String {
        let properties = schema_properties(&entry.schema);
        let required = declared_required(&entry.schema, &properties);
        let empty = Value::Object(Map::new());

        if compact {
            let args: Vec<String> = required
                .iter()
                .map(|name| {
                    let info = properties.get(name).filter(|v| truthy(v)).unwrap_or(&empty);
                    format!("{name}: {}", type_label(info))
                })
                .collect();
            return format!("[{number}] {}({})", entry.qualified, args.join(", "));
        }

        let mut head = format!("[{number}] {}", entry.qualified);
        if !entry.description.is_empty() {
            head.push_str(&format!(" — {}", Self::trim(&entry.description, MAX_TOOL_DESC)));
        }
        if required.is_empty() {
            return format!("{head}\n      (no required arguments)");
        }

        let mut lines = vec![head];
        for name in &required {
            let info = properties.get(name).filter(|v| truthy(v)).unwrap_or(&empty);
            let mut line = format!("      {name} ({})", type_label(info));
            let desc = info.get("description").and_then(Value::as_str).unwrap_or("").trim();
            if !desc.is_empty() {
                line.push_str(&format!(": {}", Self::trim(desc, MAX_PARAM_DESC)));
            }
            if let Some(Value::Array(values)) = info.get("enum") {
                if !values.is_empty() {
                    let shown: Vec<String> = values.iter().take(8).map(py_str).collect();
                    line.push_str(&format!(" [one of: {}]", shown.join(", ")));
                }
            }
            lines.push(line);
        }
        lines.join("\n")
    }

    /// Collapse whitespace and cut to `limit` chars with an ellipsis.
    pub fn trim(text: &str, limit: usize) -> String {
        static WS: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").expect("valid regex"));
        let flat = WS.replace_all(text, " ").trim().to_string();
        if flat.chars().count() <= limit {
            flat
        } else {
            take_chars(&flat, limit).trim_end().to_string() + "…"
        }
    }

    /// How many characters of catalog this model can be shown.
    ///
    /// Measured against the model actually running the call rather than a
    /// fixed number: a 13k-context model has room for 42 tools *with* their
    /// descriptions, and the descriptions are what the pick is made on — a
    /// list of bare names had a 7.5B answer "none of these apply" to a
    /// question its Notion server plainly served.  A third of the window
    /// leaves the rest for the request, the prior results and the reasoning
    /// block.  A model that reports no window (0) gets [`MAX_CATALOG_CHARS`].
    pub fn catalog_budget(&self, llm: &dyn BaseLLM) -> usize {
        let n_ctx = llm.get_max_tokens();
        if n_ctx > 0 {
            // ~3.5 chars per token holds for tool names and prose alike; a
            // rough conversion is enough for a budget.
            return MIN_CATALOG_CHARS.max((n_ctx as f64 / 3.0 * 3.5) as usize);
        }
        MAX_CATALOG_CHARS
    }

    /// The resolve prompt over `entries`.
    pub fn prompt(
        &self,
        directive: &str,
        context: Option<&str>,
        entries: &[Entry],
        llm: Option<&dyn BaseLLM>,
    ) -> String {
        // Every tool has to be in the list — a tool the model cannot see is a
        // tool it cannot pick, and dropping the tail of a 42-tool catalog means
        // the answer depends on the order the servers happened to be
        // registered in.  So the *descriptions* go first when the catalog is
        // too long, not the tools: a bare `server.tool(arg: type)` line is a
        // fifth the size and still identifies the call.  Only if even that
        // overflows are entries dropped, and then the log says how many.
        let budget = match llm {
            Some(l) => self.catalog_budget(l),
            None => MAX_CATALOG_CHARS,
        };
        let mut signatures: Vec<String> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| self.signature(e, i + 1, false))
            .collect();
        if signatures.iter().map(|b| b.chars().count()).sum::<usize>() > budget {
            tracing::info!(
                target: LOGGER,
                "[mcp-agent] {} tool(s) exceed {} chars — listing call shapes without descriptions",
                entries.len(),
                budget
            );
            signatures = entries
                .iter()
                .enumerate()
                .map(|(i, e)| self.signature(e, i + 1, true))
                .collect();
        }

        let mut used = 0usize;
        let mut kept: Vec<String> = Vec::new();
        for block in signatures {
            used += block.chars().count();
            if used > budget && !kept.is_empty() {
                tracing::warn!(
                    target: LOGGER,
                    "[mcp-agent] catalog truncated to {} of {} tool(s) at {} chars — the rest cannot be picked for this step",
                    kept.len(),
                    entries.len(),
                    budget
                );
                break;
            }
            kept.push(block);
        }

        let mut context_block = String::new();
        if let Some(ctx) = context {
            if !ctx.trim().is_empty() {
                context_block = format!(
                    "Context from earlier steps (use it to fill arguments):\n{}\n\n",
                    Self::trim(ctx, MAX_CONTEXT_CHARS)
                );
            }
        }

        format!(
            "<system>
You resolve a request into exactly ONE tool call, chosen from the numbered list.
Respond with ONLY a JSON object: {{\"tool\": <number>, \"arguments\": {{...}}}}

Rules:
- \"tool\" MUST be one of the numbers in the list, or 0. Never null, never a name,
  never a number that is not listed. No other keys, no commentary, no fences.
- The request names what the user wants, not the tool. A request about content
  a listed server holds (\"the pitch doc in Notion\", \"my open issues\") is served
  by that server's search or read tool — pick it and put the subject of the
  request in its query argument. Reading is cheap and a near miss is useful;
  answer 0 only when no listed server could hold what is being asked about.
- \"arguments\" holds that tool's listed arguments, taken from the request or the
  context. The text after an argument name is a DESCRIPTION of the field —
  guidance, never a value.
- Omit an argument whose value is not stated in the request or the context.
  Never invent an id, a cursor, a token or a URL.

Example: for \"outline the roadmap page in Notion\" with a listed
[7] Notion.notion-search(query: string), answer {{\"tool\": 7, \"arguments\":
{{\"query\": \"roadmap\"}}}}.
</system>
<user>
Request: {directive}

{context_block}Tools:
{tools}

JSON:
</user>",
            tools = kept.join("\n"),
        )
    }

    // ------------------------------------------------------------------ parsing

    /// `(index into entries, raw arguments)`, or `(None, {})`.
    ///
    /// Accepts what the models actually emit: the JSON object asked for, the
    /// same object with the tool named instead of numbered, a bare number from
    /// a model that ignored the shape, and — last — a reply that declined
    /// (`"tool": null`) while naming a tool somewhere in its own prose.  An
    /// explicit `0` is never salvaged: that is the model saying no, and it is
    /// allowed to.
    pub fn parse(&self, text: &str, entries: &[Entry]) -> (Option<usize>, Map<String, Value>) {
        static FENCE: Lazy<Regex> = Lazy::new(|| Regex::new(r"```\w*\s*").expect("valid regex"));
        let stripped = FENCE
            .replace_all(text, "")
            .trim()
            .trim_end_matches('`')
            .to_string();
        if stripped.is_empty() {
            return (None, Map::new());
        }

        let mut payload: Value = self
            .argument_validators()
            .and_then(|v| v.parse_json(&stripped).ok())
            .unwrap_or(Value::Null);

        if let Value::Array(items) = &payload {
            if let Some(first) = items.first() {
                payload = first.clone();
            }
        }

        if let Value::Object(obj) = &payload {
            let arguments = match obj.get("arguments") {
                Some(Value::Object(a)) => a.clone(),
                _ => Map::new(),
            };
            let picked = if let Some(v) = obj.get("tool") {
                v.clone()
            } else if let Some(v) = obj.get("name") {
                v.clone()
            } else {
                obj.get("tool_name").cloned().unwrap_or(Value::Null)
            };
            if let Some(index) = Self::resolve_choice(&picked, entries) {
                return (Some(index), arguments);
            }
            if is_python_zero(&picked) {
                return (None, Map::new());
            }
            // Declined without saying 0 — a small model answers `"tool": null`
            // with a note about what the tools are.  If that note names one,
            // take it: re-asking costs another reasoning pass.
            return match Self::named_in(&stripped, entries) {
                Some(index) => (Some(index), arguments),
                None => (None, Map::new()),
            };
        }

        // A bare number ("2", "Tool 2") — no arguments to be had.
        if let Some(index) = Self::resolve_choice(&Value::String(stripped.clone()), entries) {
            return (Some(index), Map::new());
        }
        match Self::named_in(&stripped, entries) {
            Some(index) => (Some(index), Map::new()),
            None => (None, Map::new()),
        }
    }

    /// The first catalog tool named anywhere in `text`, or `None`.
    ///
    /// Longest name first, so `notion-search` is not shadowed by a shorter
    /// name it contains.
    pub fn named_in(text: &str, entries: &[Entry]) -> Option<usize> {
        let low = text.to_lowercase();
        let mut best: Option<(usize, usize)> = None; // (position, index)
        for (i, entry) in entries.iter().enumerate() {
            for name in [entry.qualified.to_lowercase(), entry.tool_name.to_lowercase()] {
                if let Some(at) = low.find(&name) {
                    if best.map(|(pos, _)| at < pos).unwrap_or(true) {
                        best = Some((at, i));
                    }
                }
            }
        }
        best.map(|(_, i)| i)
    }

    /// Turn a number or a tool name into an index into `entries`.
    pub fn resolve_choice(picked: &Value, entries: &[Entry]) -> Option<usize> {
        let in_range = |number: i128| -> Option<usize> {
            if number >= 1 && number <= entries.len() as i128 {
                Some((number - 1) as usize)
            } else {
                None
            }
        };
        match picked {
            Value::Null | Value::Bool(_) => None,
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    in_range(i as i128)
                } else if let Some(u) = n.as_u64() {
                    in_range(u as i128)
                } else {
                    let f = n.as_f64()?;
                    if !f.is_finite() {
                        return None;
                    }
                    in_range(f.trunc() as i128)
                }
            }
            Value::String(s) => {
                let name = s.trim();
                if name.is_empty() {
                    return None;
                }
                let lowered = name.to_lowercase();
                for (i, entry) in entries.iter().enumerate() {
                    if lowered == entry.qualified.to_lowercase() || lowered == entry.tool_name.to_lowercase() {
                        return Some(i);
                    }
                }
                // "2", "[2]", "Tool 2: linear.create_issue" — the leading number
                // wins, but only when the reply is not mostly prose about a
                // named tool.
                static NUMBER: Lazy<Regex> = Lazy::new(|| Regex::new(r"\d+").expect("valid regex"));
                let m = NUMBER.find(name)?;
                match m.as_str().parse::<i128>() {
                    Ok(number) => in_range(number),
                    Err(_) => None,
                }
            }
            _ => None,
        }
    }

    // ---------------------------------------------------------------- arguments

    /// Keep the required arguments that carry a real value.
    ///
    /// The same filters the argument extractor applies, for the same reasons:
    /// a small model answers "MISSING", echoes the field's description back as
    /// its value, or fabricates a cursor it cannot know.  Anything the tool
    /// does not declare is dropped rather than sent.
    ///
    /// `keep_optional` keeps declared optional parameters too — a native
    /// call, whose model saw the whole schema and was told to fill an
    /// optional one only when its value is stated.
    pub fn clean_arguments(
        &self,
        schema: &Map<String, Value>,
        raw: &Map<String, Value>,
        keep_optional: bool,
    ) -> Map<String, Value> {
        if raw.is_empty() {
            return Map::new();
        }
        let properties = schema_properties(schema);
        let required: HashSet<String> = match schema.get("required") {
            Some(Value::Array(r)) => r.iter().filter_map(Value::as_str).map(str::to_string).collect(),
            _ => HashSet::new(),
        };
        let validators = self.argument_validators();
        let empty = Value::Object(Map::new());

        let mut cleaned = Map::new();
        for (key, value) in raw {
            if !properties.contains_key(key)
                || (!keep_optional && !required.is_empty() && !required.contains(key))
            {
                continue;
            }
            let placeholder = match value {
                Value::Null => true,
                Value::String(s) => matches!(s.as_str(), "" | "null" | "MISSING"),
                Value::Array(a) => a.is_empty(),
                Value::Object(o) => o.is_empty(),
                _ => false,
            };
            if placeholder {
                continue;
            }
            let info = properties.get(key).filter(|v| truthy(v)).unwrap_or(&empty);
            if let Some(v) = validators {
                if v.is_cursor_parameter(key) {
                    continue;
                }
                if v.is_schema_echo(value, info) {
                    continue;
                }
                if v.is_placeholder_echo(key, value, info) {
                    continue;
                }
            }
            cleaned.insert(key.clone(), value.clone());
        }
        cleaned
    }

    /// The extractor, used for its value checks only (never for a call).
    ///
    /// Reusing it keeps one definition of "this is a fabricated id / an echoed
    /// description" instead of a second copy that drifts from the first.
    fn argument_validators(&self) -> Option<&MCPArgumentExtractor> {
        self.validators
            .get_or_init(|| Some(MCPArgumentExtractor::new(self.llm.clone())))
            .as_ref()
    }

    // -------------------------------------------------------------------- model

    /// True when `llm` is a model that can reason.
    ///
    /// Both LLM handles answer this the same way — `LlamaCpp` from the flag
    /// bound onto its model at load (bound only for a template that reads
    /// `enable_thinking`, including a user-built or merged model the model
    /// registry says nothing about), and `RemoteLLM` by asking the host, which
    /// owns the weights.  Reported in the log line and used to route the
    /// resolve node; the call itself asks for reasoning either way.
    pub fn can_think(llm: &dyn BaseLLM) -> bool {
        llm.supports_thinking()
    }

    /// A short name for the model in a log line, never a panic.
    pub fn model_label(llm: &dyn BaseLLM) -> String {
        let label = llm.model_label();
        if !label.is_empty() {
            return label;
        }
        if let Some(model) = llm.llama_model() {
            let meta = model.metadata();
            let name = display_model_label(meta.get("general.name").map(String::as_str).unwrap_or(""));
            if !name.is_empty() {
                return name;
            }
        }
        llm.llm_type().to_string()
    }

    /// `want` tokens, or as many as fit after the prompt (minimum 64).
    ///
    /// `LlamaCpp::call` takes a requested `max_tokens` at face value, and
    /// `prompt + output` over `n_ctx` is rejected outright by llama.cpp — so a
    /// whole-catalog prompt has to ask for what is actually left.  A model that
    /// reports no window (0) cannot be measured, and is asked for `want`.
    pub fn budget(llm: &dyn BaseLLM, prompt: &str, want: usize) -> usize {
        let max_tokens = llm.get_max_tokens();
        if max_tokens == 0 {
            return want;
        }
        let headroom = max_tokens as i64 - llm.get_num_tokens(prompt) as i64 - 64;
        64i64.max((want as i64).min(headroom)) as usize
    }
}

/// Python's `picked == 0` (true for `0`, `0.0` and `False`).
fn is_python_zero(value: &Value) -> bool {
    match value {
        Value::Bool(b) => !*b,
        Value::Number(n) => n.as_f64() == Some(0.0),
        _ => false,
    }
}
