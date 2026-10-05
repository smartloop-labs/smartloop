//! One read-only MCP tool call, for a background pass that makes one.
//!
//! The account probe ([`crate::mcp::account`]) needs a tool call that changes
//! nothing, its text back, and a failure treated as a skip rather than an
//! error — including the expired-OAuth-token retry the live tool path does.
//! The 401 retry in particular is the kind of thing that gets forgotten in a
//! re-implementation and then only shows up as a background pass that silently
//! stops working an hour into a session, so it lives in one place.
//!
//! The transport underneath is
//! [`crate::agents::mcp::mcp_tool_caller::MCPToolCaller`] — the same type the
//! live agent path calls tools through, so a background call and a
//! model-issued one reach a server identically.  It is reached through the
//! [`ReadOnlyToolCaller`] trait so tests can inject a fake.  Only its
//! transport-level helpers are used here; the argument extractor and result
//! mapper it carries for the live path are LLM-backed and have no place in a
//! background pass (see [`transport_only_tool_caller`]).
//!
//! Port of `smartloop/mcp/read_only_calls.py`.

use std::any::Any;
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Map, Value};

use crate::agents::mcp::argument_extractor::MCPArgumentExtractor;
use crate::agents::mcp::mcp_tool_caller::MCPToolCaller;
use crate::agents::mcp::result_mapper::MCPResultMapper;
use crate::llms::base_llm::{BaseLLM, CallOptions, CancelEvent, ChatMessage, GenerationStream};
use crate::mcp::client::{py_truthy, take_chars, MCPToolCallResult};
use crate::workspace::MCPServer;

/// Characters kept from a failed call's reason.  Long enough to keep the part
/// of a server's validation message that names the offending field —
/// truncating before it turns the one log line that explains a failure into a
/// line that doesn't.
pub const MAX_REASON_CHARS: usize = 500;

/// Logger name used when a caller does not pass its own label's logger.
const LOGGER: &str = "smartloop.mcp.read_only_calls";

/// Outcome of one read-only call.
///
/// A struct with [`ToolCallResult::into_tuple`] so `(payload, changed, error)`
/// destructuring keeps working while the fields have names at the call sites
/// that want them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCallResult {
    pub payload: Option<String>,
    /// True when an expired token was refreshed, so the caller can persist it.
    pub credentials_changed: bool,
    pub error: Option<String>,
}

impl ToolCallResult {
    pub fn new(payload: Option<String>, credentials_changed: bool, error: Option<String>) -> Self {
        Self { payload, credentials_changed, error }
    }

    /// `(payload, credentials_changed, error)`, for tuple-style destructuring.
    pub fn into_tuple(self) -> (Option<String>, bool, Option<String>) {
        (self.payload, self.credentials_changed, self.error)
    }
}

/// What a background pass needs from the MCP transport: one tool call with the
/// expired-token refresh-and-retry applied.
///
/// Returns `(result, credentials_changed)`; an `Err` is a transport that
/// raised.  Implemented by [`MCPToolCaller`].
pub trait ReadOnlyToolCaller: Send + Sync {
    fn call_with_refresh(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
    ) -> anyhow::Result<(MCPToolCallResult, bool)>;
}

impl ReadOnlyToolCaller for MCPToolCaller {
    fn call_with_refresh(
        &self,
        server: &mut MCPServer,
        tool_name: &str,
        arguments: &Map<String, Value>,
    ) -> anyhow::Result<(MCPToolCallResult, bool)> {
        // A transport that panics is the Rust analogue of one that raised: the
        // pass must skip it rather than die with it.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            MCPToolCaller::call_with_refresh(self, server, tool_name, Some(arguments), None)
        }));
        outcome.map_err(|panic| anyhow::anyhow!(panic_message(&panic)))
    }
}

/// The text of a caught panic payload.
fn panic_message(panic: &Box<dyn Any + Send>) -> String {
    if let Some(s) = panic.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = panic.downcast_ref::<String>() {
        s.clone()
    } else {
        "tool caller panicked".to_string()
    }
}

// ------------------------------------------------------------------ JSON text

/// Python's `json.dumps(value)` with the default separators (`", "`, `": "`)
/// and `ensure_ascii=True`.
pub fn python_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_json(value, &mut out);
    out
}

fn write_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&python_number(n)),
        Value::String(s) => write_json_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_string(k, out);
                out.push_str(": ");
                write_json(v, out);
            }
            out.push('}');
        }
    }
}

/// A JSON number as Python's `repr` writes it (`1e+20`, not `1e20`).
fn python_number(n: &serde_json::Number) -> String {
    let text = n.to_string();
    if n.is_f64() {
        if let Some(pos) = text.find('e') {
            let (mantissa, exp) = text.split_at(pos);
            let exp = &exp[1..];
            if !exp.starts_with('-') && !exp.starts_with('+') {
                let exp = if exp.len() < 2 { format!("0{exp}") } else { exp.to_string() };
                return format!("{mantissa}e+{exp}");
            }
            if let Some(digits) = exp.strip_prefix('-') {
                let digits = if digits.len() < 2 { format!("0{digits}") } else { digits.to_string() };
                return format!("{mantissa}e-{digits}");
            }
        }
    }
    text
}

fn write_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{:04x}", unit));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python's `repr()` of a parsed JSON value (`'a'`, `True`, `None`, `[1, 2]`).
fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(true) => "True".to_string(),
        Value::Bool(false) => "False".to_string(),
        Value::Number(n) => python_number(n),
        Value::String(s) => {
            let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
            let mut out = String::new();
            out.push(quote);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
        Value::Array(items) => format!("[{}]", items.iter().map(python_repr).collect::<Vec<_>>().join(", ")),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{}: {}", python_repr(&Value::String(k.clone())), python_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python's `str()` of a parsed JSON value.
fn python_str(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => python_repr(other),
    }
}

// ------------------------------------------------------------------ helpers

/// Join an MCP tool result's content blocks into one string.
pub fn content_to_text(content: Option<&[Value]>) -> String {
    let Some(content) = content else {
        return String::new();
    };
    if content.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::with_capacity(content.len());
    for block in content {
        match block {
            Value::Object(obj) => {
                if obj.get("type").and_then(Value::as_str) == Some("text") {
                    // `block.get("text") or ""`
                    let text = match obj.get("text") {
                        Some(v) if py_truthy(Some(v)) => python_str(v),
                        _ => String::new(),
                    };
                    parts.push(text);
                } else {
                    parts.push(python_json_dumps(block));
                }
            }
            other => parts.push(python_str(other)),
        }
    }
    parts
        .into_iter()
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Short reason a tool call produced nothing.
///
/// MCP servers report an argument rejection as a *successful* call with
/// `is_error` set and the message in the content, so the content is read too —
/// otherwise the most common failure ("query is required") shows up as a bare
/// "no content".
pub fn failure_reason(result: &MCPToolCallResult) -> String {
    if let Some(error) = result.error.as_deref().filter(|e| !e.is_empty()) {
        return take_chars(error, MAX_REASON_CHARS);
    }
    if let Some(content) = result.content.as_deref().filter(|c| !c.is_empty()) {
        let text = content_to_text(Some(content));
        if !text.is_empty() {
            return take_chars(&text, MAX_REASON_CHARS);
        }
    }
    "empty result".to_string()
}

// ------------------------------------------------------------------ callers

/// The stand-in model of a transport-only [`MCPToolCaller`].
///
/// Python builds `MCPToolCaller(arg_extractor=None, result_mapper=None)` for a
/// background pass: the transport helpers never touch the LLM-backed argument
/// extractor.  The Rust caller requires an extractor, so it gets one backed by
/// this model, which refuses every generation — reaching it would be a bug.
#[derive(Debug, Default)]
pub struct TransportOnlyLLM {
    cancelled: Mutex<Option<CancelEvent>>,
}

impl TransportOnlyLLM {
    pub fn new() -> Self {
        Self::default()
    }
}

impl BaseLLM for TransportOnlyLLM {
    fn project_id(&self) -> &str {
        ""
    }

    fn get_num_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }

    fn apply_chat_template(&self, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        messages
    }

    fn get_max_tokens(&self) -> usize {
        0
    }

    fn call(&self, _prompt: &str, _stop: Option<&[String]>, _options: CallOptions) -> anyhow::Result<String> {
        anyhow::bail!("transport-only MCP tool caller has no language model")
    }

    fn stream<'a>(&'a self, _prompt: &str, _options: CallOptions) -> GenerationStream<'a> {
        Box::new(std::iter::empty())
    }

    fn invoke(&self, _prompt: &str, _max_tokens: Option<usize>, _thinking: Option<bool>) -> String {
        "Error generating response: transport-only MCP tool caller has no language model".to_string()
    }

    fn cancelled(&self) -> Option<CancelEvent> {
        self.cancelled.lock().clone()
    }

    fn set_cancelled(&self, event: Option<CancelEvent>) {
        *self.cancelled.lock() = event;
    }

    fn llm_type(&self) -> &str {
        "transport-only"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `MCPToolCaller(arg_extractor=None, result_mapper=None)`: a caller for the
/// transport helpers only (tool calls, refresh), no LLM behind it.
pub fn transport_only_tool_caller() -> MCPToolCaller {
    let llm: Arc<dyn BaseLLM> = Arc::new(TransportOnlyLLM::new());
    MCPToolCaller::new(
        Arc::new(MCPArgumentExtractor::new(llm)),
        Arc::new(Mutex::new(MCPResultMapper::new())),
        None,
    )
}

/// Return `existing`, or build the default transport caller lazily.
pub fn ensure_tool_caller(existing: Option<Arc<dyn ReadOnlyToolCaller>>) -> Option<Arc<dyn ReadOnlyToolCaller>> {
    if existing.is_some() {
        return existing;
    }
    match std::panic::catch_unwind(transport_only_tool_caller) {
        Ok(caller) => Some(Arc::new(caller) as Arc<dyn ReadOnlyToolCaller>),
        Err(panic) => {
            tracing::warn!(target: LOGGER, "[mcp] No MCP tool caller available: {}", panic_message(&panic));
            None
        }
    }
}

/// Call one read-only tool and return its text.
///
/// A failure is a skip, not an error: a background pass covers what it can
/// reach and the next one tries again — but the reason travels back so a pass
/// that produced nothing can say why.  An expired OAuth token is refreshed
/// once and the call retried, mirroring the live tool path.
pub fn call_read_only(
    caller: Option<&dyn ReadOnlyToolCaller>,
    server: &mut MCPServer,
    tool_name: &str,
    arguments: &Map<String, Value>,
    label: &str,
) -> ToolCallResult {
    let Some(caller) = caller else {
        return ToolCallResult::new(None, false, Some("no MCP tool caller available".to_string()));
    };
    match caller.call_with_refresh(server, tool_name, arguments) {
        Ok((result, credentials_changed)) => {
            let has_content = result.content.as_ref().is_some_and(|c| !c.is_empty());
            if !result.success || result.is_error || !has_content {
                let reason = failure_reason(&result);
                let names = arguments.keys().cloned().collect::<Vec<_>>().join(", ");
                tracing::info!(
                    target: LOGGER,
                    "[{}] {}.{}({}) returned nothing: {}",
                    label,
                    server.name,
                    tool_name,
                    if names.is_empty() { "-".to_string() } else { names },
                    reason
                );
                return ToolCallResult::new(None, credentials_changed, Some(reason));
            }
            let text = content_to_text(result.content.as_deref());
            ToolCallResult::new(if text.is_empty() { None } else { Some(text) }, credentials_changed, None)
        }
        Err(e) => {
            tracing::info!(target: LOGGER, "[{}] {}.{} raised: {}", label, server.name, tool_name, e);
            ToolCallResult::new(None, false, Some(take_chars(&e.to_string(), MAX_REASON_CHARS)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dumps_matches_python() {
        assert_eq!(python_json_dumps(&json!({"a": [1, "é"], "b": null})), r#"{"a": [1, "é"], "b": null}"#);
        assert_eq!(python_json_dumps(&json!(1e20)), "1e+20");
    }

    #[test]
    fn content_blocks_join() {
        let content = vec![json!({"type": "text", "text": "a"}), json!({"type": "image", "data": "x"}), json!("raw")];
        assert_eq!(content_to_text(Some(&content)), "a\n{\"type\": \"image\", \"data\": \"x\"}\nraw");
        assert_eq!(content_to_text(None), "");
    }
}
