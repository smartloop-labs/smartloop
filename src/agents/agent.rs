//! Generic LLM agent base.
//!
//! Port of `smartloop/agents/agent.py`.  The LangChain pieces it used
//! (`PromptTemplate`, `PydanticOutputParser`) and the `json_repair` package
//! have no Rust equivalent and are hand-ported here.

use std::marker::PhantomData;
use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde::de::DeserializeOwned;
use serde_json::{Map, Number, Value};

use crate::llms::base_llm::BaseLLM;

/// Shared handle to an LLM.
pub type SharedLLM = Arc<dyn BaseLLM + Send + Sync>;

// ====================================================================== prompt

/// LangChain `PromptTemplate` (f-string format): `{name}` placeholders are
/// substituted, `{{` / `}}` render as literal braces.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PromptTemplate {
    pub template: String,
    pub input_variables: Vec<String>,
}

impl PromptTemplate {
    /// Build a template.
    pub fn new(template: impl Into<String>, input_variables: &[&str]) -> Self {
        Self {
            template: template.into(),
            input_variables: input_variables.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Render the template with `vars` (`name -> value`).  Fails when a
    /// placeholder has no value (LangChain raises `KeyError`) or the template
    /// has an unmatched brace (Python's `str.format` raises `ValueError`).
    /// Extra variables are ignored, as `str.format` ignores unused kwargs.
    pub fn format(&self, vars: &[(&str, &str)]) -> anyhow::Result<String> {
        let lookup = |name: &str| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| *v);
        let chars: Vec<char> = self.template.chars().collect();
        let mut out = String::with_capacity(self.template.len());
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            match c {
                '{' => {
                    if chars.get(i + 1) == Some(&'{') {
                        out.push('{');
                        i += 2;
                        continue;
                    }
                    // Collect the replacement field up to the closing brace.
                    let mut j = i + 1;
                    let mut field = String::new();
                    while j < chars.len() && chars[j] != '}' {
                        if chars[j] == '{' {
                            anyhow::bail!("unexpected '{{' in field name");
                        }
                        field.push(chars[j]);
                        j += 1;
                    }
                    if j >= chars.len() {
                        anyhow::bail!("Single '{{' encountered in format string");
                    }
                    // `{name!r}` / `{name:spec}` — only the name selects the value.
                    let name = field.split(['!', ':']).next().unwrap_or("").trim();
                    match lookup(name) {
                        Some(v) => out.push_str(v),
                        None => anyhow::bail!("KeyError: '{}'", name),
                    }
                    i = j + 1;
                }
                '}' => {
                    if chars.get(i + 1) == Some(&'}') {
                        out.push('}');
                        i += 2;
                        continue;
                    }
                    anyhow::bail!("Single '}}' encountered in format string");
                }
                _ => {
                    out.push(c);
                    i += 1;
                }
            }
        }
        Ok(out)
    }
}

// ====================================================================== output parser

/// `json.loads(s, strict=False)`: like strict JSON, but control characters
/// (raw newlines, tabs) are allowed inside strings.
fn loads_non_strict(s: &str) -> Result<Value, serde_json::Error> {
    match serde_json::from_str::<Value>(s) {
        Ok(v) => Ok(v),
        Err(first) => {
            // Escape raw control characters that sit inside string literals,
            // which `strict=False` accepts, and retry.
            let mut escaped = String::with_capacity(s.len());
            let mut in_string = false;
            let mut backslash = false;
            let mut changed = false;
            for ch in s.chars() {
                if in_string {
                    if backslash {
                        backslash = false;
                        escaped.push(ch);
                        continue;
                    }
                    match ch {
                        '\\' => {
                            backslash = true;
                            escaped.push(ch);
                        }
                        '"' => {
                            in_string = false;
                            escaped.push(ch);
                        }
                        c if (c as u32) < 0x20 => {
                            changed = true;
                            escaped.push_str(&format!("\\u{:04x}", c as u32));
                        }
                        c => escaped.push(c),
                    }
                } else {
                    if ch == '"' {
                        in_string = true;
                    }
                    escaped.push(ch);
                }
            }
            if !changed {
                return Err(first);
            }
            serde_json::from_str::<Value>(&escaped)
        }
    }
}

/// LangChain `parse_partial_json`: parse a JSON string that may be missing
/// closing braces / quotes.  `None` when the text has a mismatched closer.
pub fn parse_partial_json(s: &str) -> Result<Option<Value>, serde_json::Error> {
    if let Ok(v) = loads_non_strict(s) {
        return Ok(Some(v));
    }

    let mut new_chars: Vec<String> = Vec::new();
    let mut stack: Vec<char> = Vec::new();
    let mut is_inside_string = false;
    let mut escaped = false;

    for ch in s.chars() {
        let mut new_char = ch.to_string();
        if is_inside_string {
            if ch == '"' && !escaped {
                is_inside_string = false;
            } else if ch == '\n' && !escaped {
                // Replace the newline character with the escape sequence.
                new_char = "\\n".to_string();
            } else if ch == '\\' {
                escaped = !escaped;
            } else {
                escaped = false;
            }
        } else if ch == '"' {
            is_inside_string = true;
            escaped = false;
        } else if ch == '{' {
            stack.push('}');
        } else if ch == '[' {
            stack.push(']');
        } else if ch == '}' || ch == ']' {
            if stack.last() == Some(&ch) {
                stack.pop();
            } else {
                return Ok(None);
            }
        }
        new_chars.push(new_char);
    }

    if is_inside_string {
        if escaped {
            // Remove unterminated escape character.
            new_chars.pop();
        }
        new_chars.push("\"".to_string());
    }

    stack.reverse();
    let closers: String = stack.iter().collect();

    while !new_chars.is_empty() {
        let candidate = format!("{}{}", new_chars.concat(), closers);
        if let Ok(v) = loads_non_strict(&candidate) {
            return Ok(Some(v));
        }
        new_chars.pop();
    }

    loads_non_strict(s).map(Some)
}

static ACTION_INPUT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?s)("action_input":\s*")(.*?)(")"#).unwrap());
static JSON_MARKDOWN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```(json)?(.*)").unwrap());
static UNESCAPED_QUOTE_RE: Lazy<fancy_regex::Regex> =
    Lazy::new(|| fancy_regex::Regex::new(r#"(?<!\\)""#).unwrap());

/// LangChain `_custom_parser`: escape newlines, tabs and quotes inside a
/// multiline `"action_input"` value.
fn custom_parser(multiline: &str) -> String {
    ACTION_INPUT_RE
        .replace_all(multiline, |caps: &regex::Captures| {
            let value = caps[2].replace('\n', "\\n").replace('\r', "\\r").replace('\t', "\\t");
            let value = UNESCAPED_QUOTE_RE.replace_all(&value, "\\\"").into_owned();
            format!("{}{}{}", &caps[1], value, &caps[3])
        })
        .into_owned()
}

/// LangChain `_parse_json`: strip whitespace/backticks, then partial-parse.
fn parse_json_inner(json_str: &str) -> Result<Option<Value>, serde_json::Error> {
    let stripped = json_str.trim_matches(|c| matches!(c, ' ' | '\n' | '\r' | '\t' | '`'));
    let processed = custom_parser(stripped);
    parse_partial_json(&processed)
}

/// LangChain `parse_json_markdown`: parse JSON, optionally fenced in a
/// markdown code block.
pub fn parse_json_markdown(json_string: &str) -> Result<Option<Value>, serde_json::Error> {
    match parse_json_inner(json_string) {
        Ok(v) => Ok(v),
        Err(_) => {
            let json_str = match JSON_MARKDOWN_RE.captures(json_string) {
                Some(caps) => caps.get(2).map(|m| m.as_str()).unwrap_or("").to_string(),
                None => json_string.to_string(),
            };
            parse_json_inner(&json_str)
        }
    }
}

/// LangChain `PydanticOutputParser.parse`: parse `text` as JSON into `T`.
pub fn parse_output<T: DeserializeOwned>(text: &str) -> anyhow::Result<T> {
    let parsed = match parse_json_markdown(text) {
        Ok(Some(v)) => v,
        Ok(None) | Err(_) => anyhow::bail!("Invalid json output: {}", text),
    };
    let type_name = std::any::type_name::<T>().rsplit("::").next().unwrap_or("output");
    serde_json::from_value::<T>(parsed.clone()).map_err(|e| {
        anyhow::anyhow!(
            "Failed to parse {} from completion {}. Got: {}",
            type_name,
            python_json_dumps(&parsed),
            e
        )
    })
}

// ====================================================================== json_repair

/// Python `json.dumps` (default separators, `ensure_ascii=True`).
pub fn python_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    write_py_json(value, &mut out);
    out
}

fn write_py_json(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => out.push_str(&py_number(n)),
        Value::String(s) => write_py_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_json(item, out);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (k, v)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_string(k, out);
                out.push_str(": ");
                write_py_json(v, out);
            }
            out.push('}');
        }
    }
}

/// Python's `repr(float)` / `str(int)` for a JSON number.
fn py_number(n: &Number) -> String {
    if n.is_i64() || n.is_u64() {
        return n.to_string();
    }
    let f = n.as_f64().unwrap_or(0.0);
    if f.is_finite() && f == f.trunc() && f.abs() < 1e16 {
        format!("{:.1}", f)
    } else {
        let s = format!("{}", f);
        if s.contains('e') || s.contains('.') { s } else { format!("{}.0", s) }
    }
}

fn write_py_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
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

/// Lenient recursive-descent JSON parser (the heuristics of `json_repair`).
struct RepairParser {
    chars: Vec<char>,
    pos: usize,
}

/// The kind of container a value is being parsed inside; it decides which
/// characters terminate an unquoted value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RepairCtx {
    Top,
    ObjectKey,
    ObjectValue,
    Array,
}

impl RepairParser {
    fn new(text: &str) -> Self {
        Self { chars: text.chars().collect(), pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.chars.get(self.pos + offset).copied()
    }

    /// Skip whitespace and `//`, `/* */`, `#` comments.
    fn skip_ws(&mut self) {
        loop {
            while matches!(self.peek(), Some(c) if c.is_whitespace()) {
                self.pos += 1;
            }
            match (self.peek(), self.peek_at(1)) {
                (Some('/'), Some('/')) | (Some('#'), _) => {
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                }
                (Some('/'), Some('*')) => {
                    self.pos += 2;
                    while self.pos < self.chars.len() {
                        if self.peek() == Some('*') && self.peek_at(1) == Some('/') {
                            self.pos += 2;
                            break;
                        }
                        self.pos += 1;
                    }
                }
                _ => return,
            }
        }
    }

    /// Next non-whitespace character at or after `from`.
    fn next_non_ws_from(&self, from: usize) -> Option<char> {
        self.chars[from.min(self.chars.len())..].iter().copied().find(|c| !c.is_whitespace())
    }

    /// Parse every top-level JSON value; skips prose around them.
    fn parse_all(&mut self) -> Option<Value> {
        let mut found: Vec<Value> = Vec::new();
        loop {
            // Skip anything that is not the start of a container.
            while let Some(c) = self.peek() {
                if c == '{' || c == '[' {
                    break;
                }
                self.pos += 1;
            }
            if self.pos >= self.chars.len() {
                break;
            }
            let v = self.parse_value(RepairCtx::Top);
            let empty = match &v {
                Value::Object(m) => m.is_empty(),
                Value::Array(a) => a.is_empty(),
                _ => false,
            };
            if !empty || found.is_empty() {
                found.push(v);
            }
        }
        match found.len() {
            0 => None,
            1 => found.pop(),
            _ => {
                // Drop empty containers once something real was found.
                found.retain(|v| match v {
                    Value::Object(m) => !m.is_empty(),
                    Value::Array(a) => !a.is_empty(),
                    _ => true,
                });
                if found.len() == 1 { found.pop() } else { Some(Value::Array(found)) }
            }
        }
    }

    fn parse_value(&mut self, ctx: RepairCtx) -> Value {
        self.skip_ws();
        match self.peek() {
            Some('{') => {
                self.pos += 1;
                self.parse_object()
            }
            Some('[') => {
                self.pos += 1;
                self.parse_array()
            }
            Some(q @ ('"' | '\'' | '\u{201c}' | '\u{201d}')) => {
                let close = if q == '\u{201c}' { '\u{201d}' } else { q };
                self.pos += 1;
                Value::String(self.parse_quoted(close, ctx))
            }
            Some(c) if c == '-' || c == '.' || c.is_ascii_digit() => self.parse_number(ctx),
            Some(_) => self.parse_bare(ctx),
            None => Value::String(String::new()),
        }
    }

    fn parse_object(&mut self) -> Value {
        let mut map = Map::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                Some(']') => {
                    // Mismatched closer: the object ends here.
                    break;
                }
                Some(',') | Some(';') => {
                    self.pos += 1;
                    continue;
                }
                _ => {}
            }
            // Key.
            let key = match self.peek() {
                Some(q @ ('"' | '\'' | '\u{201c}' | '\u{201d}')) => {
                    let close = if q == '\u{201c}' { '\u{201d}' } else { q };
                    self.pos += 1;
                    self.parse_quoted(close, RepairCtx::ObjectKey)
                }
                Some('{') | Some('[') => {
                    // A value where a key was expected; parse and discard.
                    let _ = self.parse_value(RepairCtx::ObjectValue);
                    continue;
                }
                _ => {
                    let start = self.pos;
                    while let Some(c) = self.peek() {
                        if c == ':' || c == ',' || c == '}' || c == '\n' {
                            break;
                        }
                        self.pos += 1;
                    }
                    let k: String = self.chars[start..self.pos].iter().collect();
                    if self.pos == start {
                        self.pos += 1;
                        continue;
                    }
                    k.trim().to_string()
                }
            };
            self.skip_ws();
            if self.peek() == Some(':') {
                self.pos += 1;
            }
            self.skip_ws();
            let value = match self.peek() {
                None => Value::String(String::new()),
                Some(',') | Some('}') => Value::String(String::new()),
                _ => self.parse_value(RepairCtx::ObjectValue),
            };
            if !key.is_empty() || !map.contains_key("") {
                map.insert(key, value);
            }
        }
        Value::Object(map)
    }

    fn parse_array(&mut self) -> Value {
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                Some('}') => {
                    // Mismatched closer: the array ends here.
                    break;
                }
                Some(',') => {
                    self.pos += 1;
                    continue;
                }
                _ => {}
            }
            let before = self.pos;
            let v = self.parse_value(RepairCtx::Array);
            if self.pos == before {
                self.pos += 1;
                continue;
            }
            items.push(v);
        }
        Value::Array(items)
    }

    /// Parse a quoted string body (the opening quote is consumed).  A quote
    /// only closes the string when what follows it looks like the end of a
    /// value; otherwise it is kept as a literal (unescaped inner quote).
    fn parse_quoted(&mut self, close: char, ctx: RepairCtx) -> String {
        let mut out = String::new();
        while let Some(c) = self.peek() {
            if c == '\\' {
                self.pos += 1;
                match self.peek() {
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    Some('r') => out.push('\r'),
                    Some('b') => out.push('\u{08}'),
                    Some('f') => out.push('\u{0c}'),
                    Some('/') => out.push('/'),
                    Some('\\') => out.push('\\'),
                    Some('"') => out.push('"'),
                    Some('\'') => out.push('\''),
                    Some('u') => {
                        let hex: String = self.chars[self.pos + 1..(self.pos + 5).min(self.chars.len())]
                            .iter()
                            .collect();
                        if hex.len() == 4 {
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                // Surrogate pairs.
                                if (0xD800..0xDC00).contains(&code)
                                    && self.peek_at(5) == Some('\\')
                                    && self.peek_at(6) == Some('u')
                                {
                                    let low: String =
                                        self.chars[self.pos + 7..(self.pos + 11).min(self.chars.len())]
                                            .iter()
                                            .collect();
                                    if let Ok(lo) = u32::from_str_radix(&low, 16) {
                                        let combined = 0x10000 + ((code - 0xD800) << 10) + (lo.wrapping_sub(0xDC00));
                                        if let Some(ch) = char::from_u32(combined) {
                                            out.push(ch);
                                            self.pos += 11;
                                            continue;
                                        }
                                    }
                                }
                                if let Some(ch) = char::from_u32(code) {
                                    out.push(ch);
                                }
                                self.pos += 5;
                                continue;
                            }
                        }
                        out.push('\\');
                        out.push('u');
                    }
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => break,
                }
                self.pos += 1;
                continue;
            }
            if c == close {
                let next = self.next_non_ws_from(self.pos + 1);
                let closes = match next {
                    None => true,
                    Some(n) => match ctx {
                        RepairCtx::ObjectKey => n == ':' || n == '}' || n == ',',
                        RepairCtx::ObjectValue => {
                            n == ',' || n == '}' || n == ']' || n == '"' || n == '\''
                        }
                        RepairCtx::Array => n == ',' || n == ']' || n == '}' || n == '"' || n == '\'',
                        RepairCtx::Top => true,
                    },
                };
                if closes {
                    self.pos += 1;
                    return out;
                }
                out.push(c);
                self.pos += 1;
                continue;
            }
            out.push(c);
            self.pos += 1;
        }
        out
    }

    fn parse_number(&mut self, ctx: RepairCtx) -> Value {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E' | '_') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let raw: String = self.chars[start..self.pos].iter().filter(|c| **c != '_').collect();
        // A number running straight into letters is really an unquoted string.
        if let Some(c) = self.peek() {
            if c.is_alphabetic() {
                self.pos = start;
                return self.parse_bare(ctx);
            }
        }
        let mut text = raw.clone();
        if text.starts_with('.') {
            text = format!("0{}", text);
        } else if text.starts_with("-.") {
            text = format!("-0{}", &text[1..]);
        }
        if text.ends_with(['.', 'e', 'E', '-', '+']) {
            text.pop();
        }
        if let Ok(i) = text.parse::<i64>() {
            return Value::Number(i.into());
        }
        if let Ok(f) = text.parse::<f64>() {
            if let Some(n) = Number::from_f64(f) {
                return Value::Number(n);
            }
        }
        Value::String(raw)
    }

    /// An unquoted literal: `true`/`false`/`null` (and Python's `True` /
    /// `False`), otherwise an unquoted string read up to the next delimiter.
    fn parse_bare(&mut self, ctx: RepairCtx) -> Value {
        for (word, value) in [
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
            ("null", Value::Null),
            ("True", Value::Bool(true)),
            ("False", Value::Bool(false)),
        ] {
            let len = word.chars().count();
            let candidate: String = self.chars[self.pos..(self.pos + len).min(self.chars.len())].iter().collect();
            let boundary = self.chars.get(self.pos + len).map(|c| !c.is_alphanumeric() && *c != '_').unwrap_or(true);
            if candidate == word && boundary {
                self.pos += len;
                return value;
            }
        }
        let start = self.pos;
        while let Some(c) = self.peek() {
            let stop = match ctx {
                RepairCtx::ObjectValue => c == ',' || c == '}' || c == '\n' || c == ']',
                RepairCtx::Array => c == ',' || c == ']' || c == '\n' || c == '}',
                RepairCtx::ObjectKey => c == ':' || c == ',' || c == '}',
                RepairCtx::Top => c == '{' || c == '[',
            };
            if stop {
                break;
            }
            self.pos += 1;
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        Value::String(s.trim().to_string())
    }
}

/// Hand-port of `json_repair.repair_json`: best-effort repair of malformed
/// JSON (missing commas/quotes/brackets, trailing commas, single quotes,
/// surrounding prose).  Returns the repaired JSON text (Python `json.dumps`
/// formatting), or `""` when nothing JSON-like could be recovered.
pub fn repair_json(text: &str) -> String {
    // Valid JSON is returned re-serialized, untouched otherwise.
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        return python_json_dumps(&v);
    }
    let mut parser = RepairParser::new(text);
    match parser.parse_all() {
        Some(v) => python_json_dumps(&v),
        None => String::new(),
    }
}

// ====================================================================== agent

/// Input to [`Agent::invoke`]: a single prompt or a list of prompts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentInput {
    Single(String),
    Many(Vec<String>),
}

impl From<&str> for AgentInput {
    fn from(s: &str) -> Self {
        AgentInput::Single(s.to_string())
    }
}

impl From<String> for AgentInput {
    fn from(s: String) -> Self {
        AgentInput::Single(s)
    }
}

impl From<Vec<String>> for AgentInput {
    fn from(v: Vec<String>) -> Self {
        AgentInput::Many(v)
    }
}

static NON_ASCII_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\x00-\x7F]+").unwrap());
static JSON_FENCE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)```json\s*").unwrap());
static FENCE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"```\s*").unwrap());

/// An agent: an LLM, a prompt template, and the structured type `T` its
/// responses parse into.
pub struct Agent<T> {
    pub llm: SharedLLM,
    pub prompt: PromptTemplate,
    return_type: PhantomData<fn() -> T>,
}

impl<T> Clone for Agent<T> {
    fn clone(&self) -> Self {
        Self { llm: self.llm.clone(), prompt: self.prompt.clone(), return_type: PhantomData }
    }
}

impl<T: DeserializeOwned> Agent<T> {
    /// Build an agent over `llm` with `prompt`.
    pub fn new(llm: SharedLLM, prompt: PromptTemplate) -> Self {
        Self { llm, prompt, return_type: PhantomData }
    }

    /// Clean LLM response by removing markdown fences and non-ASCII characters.
    pub fn clean_response(&self, text: &str) -> String {
        // Remove non-ASCII characters first
        let text = NON_ASCII_RE.replace_all(text, "");

        // Remove markdown code fences more aggressively
        let text = JSON_FENCE_RE.replace_all(&text, "");
        let text = FENCE_RE.replace_all(&text, "");
        let mut text = text.into_owned();
        if let Some(rest) = text.strip_prefix("```") {
            text = rest.to_string();
        }
        if let Some(rest) = text.strip_suffix("```") {
            text = rest.to_string();
        }

        // Remove any remaining backticks
        let text = text.replace('`', "");
        // Clean up whitespace and newlines
        let text = text.replace('\n', "");
        text.trim().to_string()
    }

    /// Run each prompt through the LLM sequentially and parse every response
    /// into `T`.  Fails with `"Failed to parse response: ..."` when a response
    /// is empty after cleaning or does not parse.
    pub fn invoke(&self, input: AgentInput) -> anyhow::Result<Vec<T>> {
        // Process inputs sequentially for robustness
        let responses: Vec<String> = match input {
            AgentInput::Single(prompt) => vec![self.llm.invoke(&prompt, None, None)],
            AgentInput::Many(prompts) => prompts.iter().map(|p| self.llm.invoke(p, None, None)).collect(),
        };

        let mut results = Vec::with_capacity(responses.len());
        for resp in responses {
            let cleaned_response = self.clean_response(&resp);
            if cleaned_response.is_empty() {
                anyhow::bail!("Failed to parse response: Empty response after cleaning");
            }
            match parse_output::<T>(&cleaned_response) {
                Ok(parsed) => results.push(parsed),
                Err(e) => anyhow::bail!("Failed to parse response: {}", e),
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_template_substitutes_and_unescapes_braces() {
        let t = PromptTemplate::new("a {x} {{lit}} b", &["x"]);
        assert_eq!(t.format(&[("x", "1")]).unwrap(), "a 1 {lit} b");
        assert!(t.format(&[]).is_err());
    }

    #[test]
    fn repair_json_matches_json_repair() {
        let cases: &[(&str, &str)] = &[
            ("", ""),
            ("no json here", ""),
            ("{\"a\": 1", "{\"a\": 1}"),
            ("{a: 1, b: \"x\"}", "{\"a\": 1, \"b\": \"x\"}"),
            ("{'a': 'b',}", "{\"a\": \"b\"}"),
            ("text before {\"a\": [1,2", "{\"a\": [1, 2]}"),
            (
                "{\"text\": \"summary\"\n \"keywords\": [\"k1\", \"k2\"]}",
                "{\"text\": \"summary\", \"keywords\": [\"k1\", \"k2\"]}",
            ),
            ("[1, 2,]", "[1, 2]"),
            ("{\"a\": True}", "{\"a\": true}"),
            ("{\"a\": \"unterminated", "{\"a\": \"unterminated\"}"),
            ("{\"a\": 1} trailing", "{\"a\": 1}"),
            ("{\"a\": \"x\" \"b\": 2}", "{\"a\": \"x\", \"b\": 2}"),
            ("{\"msg\": \"he said \"hi\" ok\"}", "{\"msg\": \"he said \\\"hi\\\" ok\"}"),
            ("{\"a\": 1, // c\n \"b\": 2}", "{\"a\": 1, \"b\": 2}"),
            ("```json\n{\"a\":1}\n```", "{\"a\": 1}"),
            ("{\"a\": [1, {\"b\": 2]", "{\"a\": [1, {\"b\": 2}]}"),
            ("{\"a\":\n\"line1\nline2\"}", "{\"a\": \"line1\\nline2\"}"),
            ("{\"a\": .5, \"c\": -2}", "{\"a\": 0.5, \"c\": -2}"),
            ("{\"a\": \"x\u{e9}\"}", "{\"a\": \"x\\u00e9\"}"),
            ("[{\"a\":1},{\"b\":2}] [3]", "[[{\"a\": 1}, {\"b\": 2}], [3]]"),
            ("{\"key\": value with spaces, \"k2\": 2}", "{\"key\": \"value with spaces\", \"k2\": 2}"),
            ("12", "12"),
            ("true", "true"),
        ];
        for (input, expected) in cases {
            assert_eq!(&repair_json(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn parse_output_handles_fences_and_partial_json() {
        #[derive(serde::Deserialize, Debug, PartialEq)]
        struct S {
            text: String,
            keywords: Vec<String>,
        }
        let s: S = parse_output("```json\n{\"text\":\"a\",\"keywords\":[\"x\"]}\n```").unwrap();
        assert_eq!(s, S { text: "a".into(), keywords: vec!["x".into()] });
        let s: S = parse_output("{\"text\":\"a\",\"keywords\":[\"x\"").unwrap();
        assert_eq!(s.keywords, vec!["x".to_string()]);
        assert!(parse_output::<S>("prefix {\"text\":\"a\",\"keywords\":[]} suffix").is_err());
        assert!(parse_output::<S>("{\"text\":1,\"keywords\":[]}").is_err());
        assert!(parse_output::<S>("").is_err());
    }
}
