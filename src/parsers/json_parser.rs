//! Port of `smartloop/parsers/json_parser.py`.
//!
//! Render JSON (`.json`/`.jsonl` files, or JSON embedded in `.txt`) into
//! GitHub-flavored Markdown.
//!
//! MarkItDown's plain-text converter accepts `application/json` but returns the
//! raw bytes verbatim — it does not structure JSON. This parser parses the JSON
//! and emits nested bullet lists, turning arrays of uniform objects into
//! Markdown tables so the content indexes as readable structure rather than one
//! blob. Malformed JSON falls back to the raw file text.

use std::io::Read;
use std::path::Path;

use serde_json::{Map, Value};

use crate::parsers::base_parser::{read_text_universal, BaseParser, ParserError, SharedLLM};
use crate::paths::expand_home;

/// Logger name used by the Python module.
const LOGGER: &str = "smartloop.parsers.json_parser";

/// Cheap sniff: does the file's first non-space char (within the first 64
/// chars) start a JSON value (`{` or `[`)? Returns false on IO/decode errors.
///
/// Only used to route `.txt` uploads; a real `.json` extension routes here
/// directly. Reads a small prefix so a large plain-text file is not fully
/// loaded just to reject it.
pub fn looks_like_json(file_path: &str) -> bool {
    let Some(head) = read_text_prefix(&expand_home(file_path), 64) else {
        return false;
    };
    match head.trim_start().chars().next() {
        Some(c) => c == '{' || c == '[',
        None => false,
    }
}

/// Python `open(path, encoding="utf-8").read(n_chars)`: the text layer decodes
/// the first buffered chunk (8 KiB), so an invalid byte inside it raises even
/// when it lies past the requested characters. `None` on IO/decode errors.
fn read_text_prefix(path: &str, n_chars: usize) -> Option<String> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut buf = Vec::with_capacity(8192);
    file.by_ref().take(8192).read_to_end(&mut buf).ok()?;
    let text = match std::str::from_utf8(&buf) {
        Ok(t) => t,
        // A multi-byte sequence cut off by the chunk boundary is not an error.
        Err(e) if e.error_len().is_none() => std::str::from_utf8(&buf[..e.valid_up_to()]).ok()?,
        Err(_) => return None,
    };
    Some(text.chars().take(n_chars).collect())
}

/// JSON / JSON Lines to Markdown parser.
pub struct JsonParser {
    pub llm: Option<SharedLLM>,
    pub file_path: String,
}

impl JsonParser {
    pub fn new(llm: Option<SharedLLM>, file_path: impl Into<String>) -> Self {
        Self { llm, file_path: file_path.into() }
    }

    /// Return `(data, is_jsonl)`; `data` is `None` when unparseable.
    pub fn load(&self, raw: &str) -> (Option<Value>, bool) {
        if let Ok(value) = serde_json::from_str::<Value>(raw) {
            return (Some(value), false);
        }
        // Try JSON Lines: one JSON value per non-empty line.
        let mut records = Vec::new();
        for line in py_splitlines(raw) {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<Value>(line) {
                Ok(value) => records.push(value),
                Err(_) => return (None, false),
            }
        }
        if records.is_empty() { (None, false) } else { (Some(Value::Array(records)), true) }
    }

    // -- markdown rendering -------------------------------------------------

    /// Render any JSON value to Markdown at the given indent level.
    pub fn to_markdown(&self, value: &Value, indent: usize) -> String {
        match value {
            Value::Object(obj) => self.render_dict(obj, indent),
            Value::Array(arr) => self.render_list(arr, indent),
            _ => format!("{}{}", Self::pad(indent), Self::scalar(value)),
        }
    }

    fn render_dict(&self, obj: &Map<String, Value>, indent: usize) -> String {
        let mut lines = Vec::new();
        let pad = Self::pad(indent);
        for (key, val) in obj {
            match val {
                Value::Array(arr) if Self::is_table(arr) => {
                    // A table must sit at the left margin with a blank line before
                    // and after it, or Markdown renders the pipes as literal text
                    // instead of a table. Keep the key as a label above it.
                    lines.push(format!("{pad}- **{key}**:"));
                    lines.push(String::new());
                    lines.push(self.render_table(arr));
                    lines.push(String::new());
                }
                Value::Object(o) if !o.is_empty() => {
                    lines.push(format!("{pad}- **{key}**:"));
                    lines.push(self.to_markdown(val, indent + 1));
                }
                Value::Array(a) if !a.is_empty() => {
                    lines.push(format!("{pad}- **{key}**:"));
                    lines.push(self.to_markdown(val, indent + 1));
                }
                _ => lines.push(format!("{pad}- **{key}**: {}", Self::scalar(val))),
            }
        }
        lines.join("\n")
    }

    fn render_list(&self, arr: &[Value], indent: usize) -> String {
        // An array of uniform objects (no nested containers) renders as a table.
        if Self::is_table(arr) {
            return self.render_table(arr);
        }

        let mut lines = Vec::new();
        let pad = Self::pad(indent);
        for item in arr {
            if is_nonempty_container(item) {
                lines.push(format!("{pad}-"));
                lines.push(self.to_markdown(item, indent + 1));
            } else {
                lines.push(format!("{pad}- {}", Self::scalar(item)));
            }
        }
        lines.join("\n")
    }

    /// True when `arr` is non-empty, all objects, and every cell is a scalar.
    pub fn is_table(arr: &[Value]) -> bool {
        if arr.is_empty() || !arr.iter().all(Value::is_object) {
            return false;
        }
        // Every cell must be a scalar so rows stay on one line.
        arr.iter()
            .filter_map(Value::as_object)
            .flat_map(|row| row.values())
            .all(|v| !v.is_object() && !v.is_array())
    }

    fn render_table(&self, arr: &[Value]) -> String {
        // Rendered at the left margin (no indent): an indented pipe-table is
        // not recognized as a table by GitHub-flavored Markdown.
        let mut columns: Vec<&String> = Vec::new();
        for row in arr.iter().filter_map(Value::as_object) {
            for key in row.keys() {
                if !columns.contains(&key) {
                    columns.push(key);
                }
            }
        }

        let header: Vec<&str> = columns.iter().map(|c| c.as_str()).collect();
        let mut lines = vec![
            format!("| {} |", header.join(" | ")),
            format!("| {} |", vec!["---"; columns.len()].join(" | ")),
        ];
        let empty = Value::String(String::new());
        for row in arr.iter().filter_map(Value::as_object) {
            let cells: Vec<String> = columns.iter().map(|col| self.cell(row.get(*col).unwrap_or(&empty))).collect();
            lines.push(format!("| {} |", cells.join(" | ")));
        }
        lines.join("\n")
    }

    fn pad(indent: usize) -> String {
        "  ".repeat(indent)
    }

    /// Python `str()` of a scalar: `null`, `true`/`false`, numbers, raw strings.
    pub fn scalar(value: &Value) -> String {
        match value {
            Value::Null => "null".to_string(),
            Value::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            Value::Number(n) => py_number_str(n),
            Value::String(s) => s.clone(),
            // Containers only reach here when empty (`{}` / `[]`), which
            // Python prints with `str()`.
            Value::Array(a) if a.is_empty() => "[]".to_string(),
            Value::Object(o) if o.is_empty() => "{}".to_string(),
            other => py_repr_json(other),
        }
    }

    /// Table cell: scalar with `|` escaped and newlines replaced by spaces.
    fn cell(&self, value: &Value) -> String {
        // Escape pipes so a value never breaks the table layout.
        Self::scalar(value).replace('|', "\\|").replace('\n', " ")
    }
}

impl BaseParser for JsonParser {
    fn llm(&self) -> Option<&SharedLLM> {
        self.llm.as_ref()
    }
    fn file_path(&self) -> &str {
        &self.file_path
    }
    fn type_name(&self) -> &'static str {
        "JsonParser"
    }
    fn parse(&self) -> anyhow::Result<String> {
        let filepath = expand_home(&self.file_path);
        if !Path::new(&filepath).exists() {
            return Err(ParserError::FileNotFound(filepath).into());
        }

        let raw = read_text_universal(&filepath)?;

        let (data, is_jsonl) = self.load(&raw);
        let Some(mut data) = data else {
            // Not valid JSON after all — index the raw text rather than fail.
            tracing::info!(target: LOGGER, "JsonParser: {} is not valid JSON, indexing as raw text", filepath);
            return Ok(raw.trim().to_string());
        };

        if is_jsonl {
            let mut wrapped = Map::new();
            wrapped.insert("records".to_string(), data);
            data = Value::Object(wrapped);
        }
        Ok(self.to_markdown(&data, 0).trim().to_string())
    }
}

fn is_nonempty_container(value: &Value) -> bool {
    match value {
        Value::Object(o) => !o.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    }
}

/// Python `str()` of a JSON number: ints verbatim, floats via `repr`.
fn py_number_str(n: &serde_json::Number) -> String {
    if n.is_i64() || n.is_u64() {
        n.to_string()
    } else {
        n.as_f64().map(py_float_repr).unwrap_or_else(|| n.to_string())
    }
}

/// Python `repr()` of a JSON container (dict/list of Python objects).
pub fn py_repr_json(value: &Value) -> String {
    match value {
        Value::Null => "None".to_string(),
        Value::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Value::Number(n) => py_number_str(n),
        Value::String(s) => py_str_repr(s),
        Value::Array(a) => format!("[{}]", a.iter().map(py_repr_json).collect::<Vec<_>>().join(", ")),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", py_str_repr(k), py_repr_json(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// Python `repr()` of a `str`.
fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
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
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr(float)`: shortest round-trip digits, fixed notation for
/// exponents in `[-4, 16)` (always with a fractional part), otherwise
/// scientific notation with a signed, at-least-two-digit exponent.
pub fn py_float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }

    // `{:e}` yields the shortest round-trip digits, e.g. "1.2345e-7".
    let sci = format!("{value:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();

    let body = if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                format!("{}{}.0", digits, "0".repeat(int_len - digits.len()))
            } else {
                format!("{}.{}", &digits[..int_len], &digits[int_len..])
            }
        } else {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        }
    } else {
        let mant = if digits.len() > 1 { format!("{}.{}", &digits[..1], &digits[1..]) } else { digits.clone() };
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mant}e{sign}{:02}", exp.abs())
    };

    if negative { format!("-{body}") } else { body }
}

/// Python `str.splitlines()`: splits on `\n`, `\r`, `\r\n`, `\v`, `\f`,
/// `\x1c`-`\x1e`, `\x85`, `\u2028` and `\u2029`, dropping the separators.
pub fn py_splitlines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
        );
        if is_break {
            lines.push(&text[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && matches!(chars.peek(), Some((_, '\n'))) {
                chars.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        assert_eq!(py_float_repr(1.0), "1.0");
        assert_eq!(py_float_repr(0.1), "0.1");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(1e15), "1000000000000000.0");
        assert_eq!(py_float_repr(1e-5), "1e-05");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(-2.5), "-2.5");
        assert_eq!(py_float_repr(123.456), "123.456");
        assert_eq!(py_float_repr(1.5e300), "1.5e+300");
    }

    #[test]
    fn splitlines_matches_python() {
        assert_eq!(py_splitlines("a\nb\r\nc\rd\n"), vec!["a", "b", "c", "d"]);
        assert_eq!(py_splitlines(""), Vec::<&str>::new());
        assert_eq!(py_splitlines("\n\nx"), vec!["", "", "x"]);
    }

    #[test]
    fn renders_tables_and_nested_lists() {
        let parser = JsonParser::new(None, "unused.json");
        let value: Value = serde_json::from_str(
            r#"{"name": "x", "items": [{"a": 1, "b": "p|q"}, {"a": 2.5}], "tags": ["t1", {"k": null}], "empty": []}"#,
        )
        .unwrap();
        let md = parser.to_markdown(&value, 0);
        assert_eq!(
            md,
            "- **name**: x\n- **items**:\n\n| a | b |\n| --- | --- |\n| 1 | p\\|q |\n| 2.5 |  |\n\n- **tags**:\n  - t1\n  -\n    - **k**: null\n- **empty**: []"
        );
    }

    #[test]
    fn loads_json_lines() {
        let parser = JsonParser::new(None, "unused.json");
        let (data, is_jsonl) = parser.load("{\"a\": 1}\n\n{\"a\": 2}\n");
        assert!(is_jsonl);
        assert_eq!(data.unwrap(), serde_json::json!([{"a": 1}, {"a": 2}]));
        let (data, is_jsonl) = parser.load("not json\n");
        assert!(data.is_none());
        assert!(!is_jsonl);
    }
}
