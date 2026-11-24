//! MiniMax M2: `<minimax:tool_call><invoke name="bash"><parameter
//! name="command">ls</parameter></invoke></minimax:tool_call>`.  Its control
//! tokens, `]<]minimax[>[` and the like, show as text around a reply and go.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use super::{NativeToolCall, ToolCallParser};

/// MiniMax's native calls.
pub struct MiniMaxParser;

static BLOCK: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<minimax:tool_call>.*?(?:</minimax:tool_call>|$)").expect("valid regex"));
static INVOKE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)<invoke\s+name="([^"]+)"\s*>(.*?)(?:</invoke>|$)"#).expect("valid regex"));
static PARAMETER: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)<parameter\s+name="([^"]+)"\s*>(.*?)</parameter>"#).expect("valid regex"));
static CONTROL_TOKEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\]<\][A-Za-z0-9_]*\[>\[").expect("valid regex"));

impl ToolCallParser for MiniMaxParser {
    fn parse(&self, text: &str) -> Vec<NativeToolCall> {
        INVOKE
            .captures_iter(text)
            .filter_map(|c| {
                let name = c[1].trim().to_string();
                if name.is_empty() {
                    return None;
                }
                let mut arguments = Map::new();
                for p in PARAMETER.captures_iter(&c[2]) {
                    let raw = p[2].trim();
                    let value = serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
                    arguments.insert(p[1].trim().to_string(), value);
                }
                Some(NativeToolCall { id: None, name, arguments })
            })
            .collect()
    }

    fn strip(&self, text: &str) -> String {
        let text = BLOCK.replace_all(text, "");
        let text = INVOKE.replace_all(&text, "");
        CONTROL_TOKEN.replace_all(&text, "").into_owned()
    }

    fn markers(&self) -> &'static [&'static str] {
        &["<minimax:tool_call>", "<invoke ", "]<]"]
    }
}
