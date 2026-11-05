//! Native tool calling across the LLM seam.
//!
//! MCP resolution normally shows the model a numbered catalog in its prompt
//! and parses a JSON answer.  A model that calls functions natively (its chat
//! template renders a `tools` list, or its API takes one, and it was trained
//! to answer with a structured call) does better with the real thing: full
//! JSON schemas, its own call format, optional arguments it can fill.
//!
//! The tool list rides *inside the prompt string* as one leading
//! `<slp_tools>[…]</slp_tools>` block, so it crosses every hop a prompt
//! already crosses — the model host's `Generate` RPC, `ModelService`, the
//! capability router — without a new field on any of them.  Whoever owns the
//! weights lifts it back out ([`split_tools`]) and hands it to the model the
//! native way: as the `tools` variable of the chat template (`LlamaCpp`) or
//! the `tools` field of the request (`CloudLLM`).
//!
//! The answer comes back as text in one canonical shape —
//! `<tool_call>{"name": …, "arguments": {…}}</tool_call>` — which is what the
//! Qwen/Hermes family emits on its own and what `CloudLLM` renders a
//! structured `tool_calls` reply into.  [`parse_tool_calls`] also reads the
//! XML-style `<function=…><parameter=…>` calls Qwen3.5 and Qwen3-Coder emit.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};

const TOOLS_OPEN: &str = "<slp_tools>";
const TOOLS_CLOSE: &str = "</slp_tools>";
const MESSAGES_OPEN: &str = "<slp_messages>";
const MESSAGES_CLOSE: &str = "</slp_messages>";

/// One call a model asked for.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NativeToolCall {
    /// The provider's id for the call, when it gave one; a tool result
    /// answers it by this id.
    pub id: Option<String>,
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// `prompt` with `tools` (OpenAI function definitions) attached in front.
pub fn with_tools(prompt: &str, tools: &[Value]) -> String {
    format!("{TOOLS_OPEN}{}{TOOLS_CLOSE}\n{prompt}", Value::Array(tools.to_vec()))
}

/// The prompt without its leading tools block, and the tools it carried.
///
/// Only a block at the very start counts, so a user message that happens to
/// contain the tag is left alone.  A block that does not parse is dropped
/// with a warning and the prompt goes ahead without tools.
pub fn split_tools(prompt: &str) -> (&str, Option<Vec<Value>>) {
    let trimmed = prompt.trim_start();
    let Some(rest) = trimmed.strip_prefix(TOOLS_OPEN) else {
        return (prompt, None);
    };
    let Some(end) = rest.find(TOOLS_CLOSE) else {
        return (prompt, None);
    };
    let body = rest[end + TOOLS_CLOSE.len()..].trim_start_matches('\n');
    match serde_json::from_str::<Value>(&rest[..end]) {
        Ok(Value::Array(tools)) if !tools.is_empty() => (body, Some(tools)),
        Ok(_) => (body, None),
        Err(e) => {
            tracing::warn!(target: "smartloop.llms", "[tools] unreadable tools block, ignoring it: {e}");
            (body, None)
        }
    }
}

/// `calls` in the canonical text shape [`parse_tool_calls`] reads.
pub fn render_tool_calls(calls: &[NativeToolCall]) -> String {
    calls
        .iter()
        .map(|c| {
            let mut call = json!({"name": c.name, "arguments": c.arguments});
            if let Some(id) = &c.id {
                call["id"] = json!(id);
            }
            format!("<tool_call>\n{call}\n</tool_call>")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `tools` and a whole conversation (OpenAI-shaped messages: `system`,
/// `user`, `assistant` with `tool_calls`, `tool` results) as one prompt.
/// The messages are sent as they are, in place of the prompt's own tags.
pub fn with_tools_and_messages(tools: &[Value], messages: &[Value]) -> String {
    format!(
        "{TOOLS_OPEN}{}{TOOLS_CLOSE}\n{MESSAGES_OPEN}{}{MESSAGES_CLOSE}",
        Value::Array(tools.to_vec()),
        Value::Array(messages.to_vec())
    )
}

/// The conversation a prompt carries after its tools block (see
/// [`with_tools_and_messages`]), if any.  `prompt` is what [`split_tools`]
/// left.
pub fn split_messages(prompt: &str) -> Option<Vec<Value>> {
    let rest = prompt.trim_start().strip_prefix(MESSAGES_OPEN)?;
    let end = rest.find(MESSAGES_CLOSE)?;
    match serde_json::from_str::<Value>(&rest[..end]) {
        Ok(Value::Array(messages)) if !messages.is_empty() => Some(messages),
        _ => {
            tracing::warn!(target: "smartloop.llms", "[tools] unreadable messages block, ignoring it");
            None
        }
    }
}

/// An assistant turn that made `calls`, for the conversation: arguments as
/// objects (what chat templates iterate); [`openai_messages`] turns them
/// into the strings an OpenAI endpoint wants.
pub fn assistant_tool_turn(text: &str, calls: &[NativeToolCall]) -> Value {
    json!({
        "role": "assistant",
        "content": text,
        "tool_calls": calls.iter().map(|c| json!({
            "id": c.id.clone().unwrap_or_default(),
            "type": "function",
            "function": {"name": c.name, "arguments": c.arguments},
        })).collect::<Vec<_>>(),
    })
}

/// A tool's result, answering call `id`.
pub fn tool_result_turn(id: &str, name: &str, content: &str) -> Value {
    json!({"role": "tool", "tool_call_id": id, "name": name, "content": content})
}

/// `messages` as an OpenAI endpoint takes them: tool-call arguments as JSON
/// strings.
pub fn openai_messages(messages: &[Value]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let mut m = m.clone();
            if let Some(Value::Array(calls)) = m.get_mut("tool_calls") {
                for call in calls.iter_mut() {
                    if let Some(args) = call.pointer_mut("/function/arguments") {
                        if !args.is_string() {
                            *args = Value::String(args.to_string());
                        }
                    }
                }
            }
            m
        })
        .collect()
}

/// Whether a Jinja chat template renders a `tools` variable — the mark of a
/// model trained to call functions natively.
pub fn template_renders_tools(template: &str) -> bool {
    static USES_TOOLS: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"\{[%{][^%}]*\btools\b").expect("valid regex"));
    USES_TOOLS.is_match(template)
}

/// Every tool call in a model's reply, in order; empty when it made none.
///
/// Reads, in this order:
/// * `<tool_call>{"name": …, "arguments": …}</tool_call>` (Qwen2.5/Qwen3,
///   Hermes, and what `CloudLLM` renders);
/// * `<function=name><parameter=key>value</parameter></function>`
///   (Qwen3.5 / Qwen3-Coder), inside `<tool_call>` or bare;
/// * a bare `{"name": …, "arguments": …}` object as the whole reply.
pub fn parse_tool_calls(text: &str) -> Vec<NativeToolCall> {
    static TOOL_CALL: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<tool_call>(.*?)(?:</tool_call>|$)").expect("valid regex"));
    static FUNCTION: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<function=([^>\s]+)>(.*?)(?:</function>|$)").expect("valid regex"));
    static PARAMETER: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<parameter=([^>\s]+)>(.*?)</parameter>").expect("valid regex"));

    let mut calls = Vec::new();
    let blocks: Vec<&str> = TOOL_CALL
        .captures_iter(text)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    let sources: Vec<&str> = if blocks.is_empty() { vec![text] } else { blocks };

    for block in sources {
        if let Some(call) = json_call(block) {
            calls.push(call);
            continue;
        }
        for f in FUNCTION.captures_iter(block) {
            let name = f[1].trim().to_string();
            let mut arguments = Map::new();
            for p in PARAMETER.captures_iter(&f[2]) {
                let raw = p[2].trim();
                // A parameter's value is JSON when it parses as such (numbers,
                // booleans, arrays, objects), otherwise the literal text.
                let value = serde_json::from_str::<Value>(raw).unwrap_or_else(|_| Value::String(raw.to_string()));
                arguments.insert(p[1].trim().to_string(), value);
            }
            if !name.is_empty() {
                calls.push(NativeToolCall { id: None, name, arguments });
            }
        }
    }
    calls
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_block_round_trips_and_only_leads() {
        let tools = vec![json!({"type": "function", "function": {"name": "x"}})];
        let prompt = with_tools("<user>hi</user>", &tools);
        let (body, got) = split_tools(&prompt);
        assert_eq!(body, "<user>hi</user>");
        assert_eq!(got, Some(tools));

        let inner = "<user>see <slp_tools>[1]</slp_tools></user>";
        assert_eq!(split_tools(inner), (inner, None));
    }

    #[test]
    fn parses_json_tool_call() {
        let calls = parse_tool_calls(
            "<tool_call>\n{\"name\": \"create_issue\", \"arguments\": {\"title\": \"Bug\", \"priority\": 2}}\n</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "create_issue");
        assert_eq!(calls[0].arguments["priority"], json!(2));
    }

    #[test]
    fn parses_xml_function_call() {
        let text = "<tool_call>\n<function=create_issue>\n<parameter=title>\nLogin broken\n</parameter>\n<parameter=labels>\n[\"bug\"]\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(text);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].arguments["title"], json!("Login broken"));
        assert_eq!(calls[0].arguments["labels"], json!(["bug"]));
    }

    #[test]
    fn parses_openai_string_arguments_and_bare_object() {
        let calls = parse_tool_calls(r#"{"name": "search", "arguments": "{\"query\": \"roadmap\"}"}"#);
        assert_eq!(calls[0].arguments["query"], json!("roadmap"));
        assert!(parse_tool_calls("I can't help with that.").is_empty());
    }

    #[test]
    fn rendered_calls_parse_back() {
        let mut arguments = Map::new();
        arguments.insert("q".into(), json!("x"));
        let call = NativeToolCall { id: Some("call_1".into()), name: "search".into(), arguments };
        assert_eq!(parse_tool_calls(&render_tool_calls(&[call.clone()])), vec![call]);
    }

    #[test]
    fn a_conversation_round_trips_and_goes_to_openai_with_string_arguments() {
        let tools = vec![json!({"type": "function", "function": {"name": "x"}})];
        let mut arguments = Map::new();
        arguments.insert("q".into(), json!("chris"));
        let call = NativeToolCall { id: Some("c1".into()), name: "list_users".into(), arguments };
        let messages = vec![
            json!({"role": "user", "content": "assign it to chris"}),
            assistant_tool_turn("", &[call]),
            tool_result_turn("c1", "list_users", "{\"users\":[]}"),
        ];
        let prompt = with_tools_and_messages(&tools, &messages);
        let (rest, got_tools) = split_tools(&prompt);
        assert_eq!(got_tools, Some(tools));
        assert_eq!(split_messages(rest), Some(messages.clone()));

        let openai = openai_messages(&messages);
        assert_eq!(openai[1]["tool_calls"][0]["function"]["arguments"], json!("{\"q\":\"chris\"}"));
        assert_eq!(messages[1]["tool_calls"][0]["function"]["arguments"], json!({"q": "chris"}));
    }

    #[test]
    fn detects_tool_aware_templates() {
        assert!(template_renders_tools("{%- if tools %}<tools>{{ tools | tojson }}{% endif %}"));
        assert!(template_renders_tools("{%- for tool in tools %}"));
        assert!(!template_renders_tools("{% for message in messages %}{{ message.content }}{% endfor %}"));
    }
}
