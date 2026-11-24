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
//! XML-style `<function=…><parameter=…>` calls Qwen3.5 and Qwen3-Coder emit,
//! and Gemma 4's `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`.
//! Each model family's markup has its own [`ToolCallParser`]:
//! [`QwenParser`], [`GemmaParser`] and [`MiniMaxParser`].  Whichever wrote
//! it, a reply's calls leave the server in one shape (OpenAI `tool_calls`),
//! so a client reads one format.

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Map, Value};

const TOOLS_OPEN: &str = "<slp_tools>";
const TOOLS_CLOSE: &str = "</slp_tools>";
const MESSAGES_OPEN: &str = "<slp_messages>";
const MESSAGES_CLOSE: &str = "</slp_messages>";

mod gemma;
mod minimax;
mod qwen;

pub use gemma::GemmaParser;
pub use minimax::MiniMaxParser;
pub use qwen::QwenParser;

/// One model family's way of writing the calls it makes into its reply.
pub trait ToolCallParser: Sync {
    /// The calls in `text` written this way, in order.
    fn parse(&self, text: &str) -> Vec<NativeToolCall>;
    /// `text` without its calls (and control tokens) written this way.
    fn strip(&self, text: &str) -> String;
    /// What opens a call written this way: held back while a reply streams.
    fn markers(&self) -> &'static [&'static str];
}

/// The parsers, in the order tried.  Gemma's goes before Qwen's, whose bare
/// JSON reading would take a Gemma call's braces for an object; MiniMax's
/// before Qwen's for the same reason.
pub static PARSERS: [&dyn ToolCallParser; 3] = [&GemmaParser, &MiniMaxParser, &QwenParser];

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

/// `text` without the tool calls it holds, in any format: what the model
/// said around the calls it made.
pub fn strip_tool_calls(text: &str) -> String {
    PARSERS.iter().fold(text.to_string(), |text, parser| parser.strip(&text)).trim().to_string()
}

/// Every tool call in a model's reply, in order; empty when it made none.
///
/// Each parser is tried in [`PARSERS`] order and the first that finds a call
/// answers: a reply is written in one model's markup.
pub fn parse_tool_calls(text: &str) -> Vec<NativeToolCall> {
    PARSERS.iter().map(|parser| parser.parse(text)).find(|calls| !calls.is_empty()).unwrap_or_default()
}

/// The text that opens a call in any model's markup: what a stream holds
/// back until it knows whether a call follows.
pub fn call_markers() -> impl Iterator<Item = &'static str> {
    PARSERS.iter().flat_map(|parser| parser.markers().iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimax_calls_parse_and_control_tokens_go() {
        let reply = "]<]minimax[>[Let me look.\n<minimax:tool_call>\n<invoke name=\"bash\">\n<parameter name=\"command\">git status</parameter>\n<parameter name=\"timeout\">30</parameter>\n</invoke>\n</minimax:tool_call>";
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"command": "git status", "timeout": 30}));
        assert_eq!(strip_tool_calls(reply), "Let me look.");
    }

    #[test]
    fn minimax_control_tokens_around_a_json_call() {
        let reply = "]<]minimax[>[<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}</tool_call>]<]minimax[>[";
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "bash");
        assert_eq!(strip_tool_calls(reply), "");
    }

    #[test]
    fn minimax_control_token_inside_the_call_block() {
        // As MiniMax wrote it in docent, with no tools offered.
        let reply = "Let me first explore the codebase.]<]minimax[>[<tool_call> {\"name\": \"bash\", \"arguments\": {\"command\": \"ls -la /tmp/boxy-bird/\"}}\n]<]minimax[>[</tool_call>\n";
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(Value::Object(calls[0].arguments.clone()), json!({"command": "ls -la /tmp/boxy-bird/"}));
        assert_eq!(strip_tool_calls(reply), "Let me first explore the codebase.");
    }

    #[test]
    fn gemma_calls_parse() {
        let reply = "Let me look.\n<|tool_call>call:explore{paths:[<|\"|>src/lib/theme.ts<|\"|>,<|\"|>src/components/CpuTile.tsx<|\"|>]}<tool_call|>";
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "explore");
        assert_eq!(
            Value::Object(calls[0].arguments.clone()),
            json!({"paths": ["src/lib/theme.ts", "src/components/CpuTile.tsx"]})
        );
        assert_eq!(strip_tool_calls(reply), "Let me look.");
    }

    #[test]
    fn gemma_strings_keep_what_looks_like_markup() {
        let reply = "<|tool_call>call:bash{command:<|\"|>grep -n \"a, b:\" {x} file<|\"|>,timeout:30,dry:false}<tool_call|>\
                     <|tool_call>call:explore{paths:[]}<tool_call|>";
        let calls = parse_tool_calls(reply);
        assert_eq!(calls.len(), 2);
        assert_eq!(
            Value::Object(calls[0].arguments.clone()),
            json!({"command": "grep -n \"a, b:\" {x} file", "timeout": 30, "dry": false})
        );
        assert_eq!(calls[1].name, "explore");
    }

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
