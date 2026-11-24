//! Gemma 4: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`, the
//! arguments an object with bare keys and strings quoted `<|"|>…<|"|>`.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use super::{NativeToolCall, ToolCallParser};

/// Gemma 4's native calls.
pub struct GemmaParser;

/// The string delimiter inside a call's arguments.
const QUOTE: &str = "<|\"|>";

static CALL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<\|tool_call>\s*call:([^\s{<]+)\s*(\{.*?)\s*(?:<tool_call\|>|$)").expect("valid regex")
});
static BLOCK: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<\|tool_call>.*?(?:<tool_call\|>|$)").expect("valid regex"));
static BARE_KEY: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:").expect("valid regex"));

impl ToolCallParser for GemmaParser {
    /// A call whose arguments don't read is skipped.
    fn parse(&self, text: &str) -> Vec<NativeToolCall> {
        CALL.captures_iter(text)
            .filter_map(|c| {
                let Some(Value::Object(arguments)) = arguments(&c[2]) else {
                    return None;
                };
                Some(NativeToolCall { id: None, name: c[1].trim().to_string(), arguments })
            })
            .collect()
    }

    fn strip(&self, text: &str) -> String {
        BLOCK.replace_all(text, "").into_owned()
    }

    fn markers(&self) -> &'static [&'static str] {
        &["<|tool_call>"]
    }
}

/// The arguments as JSON: each `<|"|>…<|"|>` string quoted as JSON, each
/// bare key outside a string quoted.
fn arguments(text: &str) -> Option<Value> {
    let mut json = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(QUOTE) {
        json.push_str(&BARE_KEY.replace_all(&rest[..open], "$1\"$2\":"));
        let after = &rest[open + QUOTE.len()..];
        let close = after.find(QUOTE)?;
        json.push_str(&serde_json::to_string(&after[..close]).ok()?);
        rest = &after[close + QUOTE.len()..];
    }
    json.push_str(&BARE_KEY.replace_all(rest, "$1\"$2\":"));
    serde_json::from_str(&json).ok()
}
