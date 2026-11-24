//! Qwen's calls, in both the styles the family writes:
//!
//! * `<tool_call>{"name": …, "arguments": {…}}</tool_call>`: Qwen2.5/Qwen3
//!   (the Hermes format), and the shape `CloudLLM` renders a structured
//!   reply into; a bare `{"name", "arguments"}` object as the whole reply
//!   reads too;
//! * `<function=name><parameter=key>value</parameter></function>`: Qwen3.5
//!   and Qwen3-Coder, inside `<tool_call>` or bare.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use super::{NativeToolCall, ToolCallParser};

/// Qwen-family tool calls.
pub struct QwenParser;

static BLOCK: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<tool_call>(.*?)(?:</tool_call>|$)").expect("valid regex"));
static FUNCTION: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<function=([^>\s]+)>(.*?)(?:</function>|$)").expect("valid regex"));
static PARAMETER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<parameter=([^>\s]+)>(.*?)</parameter>").expect("valid regex"));

impl ToolCallParser for QwenParser {
    /// Each `<tool_call>` block (or, with none, the whole reply) as a JSON
    /// call, else as the `<function=…>` calls it holds.
    fn parse(&self, text: &str) -> Vec<NativeToolCall> {
        let blocks: Vec<&str> = BLOCK.captures_iter(text).filter_map(|c| c.get(1).map(|m| m.as_str())).collect();
        let sources = if blocks.is_empty() { vec![text] } else { blocks };
        let mut calls = Vec::new();
        for block in sources {
            match json_call(block) {
                Some(call) => calls.push(call),
                None => calls.extend(function_calls(block)),
            }
        }
        calls
    }

    fn strip(&self, text: &str) -> String {
        FUNCTION.replace_all(&BLOCK.replace_all(text, ""), "").into_owned()
    }

    fn markers(&self) -> &'static [&'static str] {
        &["<tool_call>", "<function="]
    }
}

/// The `<function=…>` calls in `text`.
fn function_calls(text: &str) -> Vec<NativeToolCall> {
    FUNCTION
        .captures_iter(text)
        .filter_map(|f| {
            let name = f[1].trim().to_string();
            if name.is_empty() {
                return None;
            }
            let mut arguments = Map::new();
            for p in PARAMETER.captures_iter(&f[2]) {
                let raw = p[2].trim();
                // A parameter's value is JSON when it parses as such (numbers,
                // booleans, arrays, objects), otherwise the literal text.
                let value = serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
                arguments.insert(p[1].trim().to_string(), value);
            }
            Some(NativeToolCall { id: None, name, arguments })
        })
        .collect()
}

/// A `{"name", "arguments"}` object (arguments possibly a JSON string, as
/// OpenAI sends them), with surrounding text and fences tolerated.
fn json_call(text: &str) -> Option<NativeToolCall> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    let value: Value = serde_json::from_str(&text[start..=end]).ok()?;
    let outer = value.as_object()?;
    let id = outer.get("id").and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    let object = match outer.get("function").and_then(Value::as_object) {
        Some(function) => function,
        None => outer,
    };
    let name = object.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = match object.get("arguments").or_else(|| object.get("parameters")) {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        },
        _ => Map::new(),
    };
    Some(NativeToolCall { id, name, arguments })
}
