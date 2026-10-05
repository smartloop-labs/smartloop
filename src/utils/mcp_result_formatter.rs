//! Port of `smartloop/utils/mcp_result_formatter.py`.

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use crate::llms::BaseLLM;
use crate::utils::py_compat::{py_repr, py_str};

/// Keys whose values carry no meaning for the LLM.
const NOISY_KEYS: &[&str] = &[
    "id", "uid", "uuid", "guid", "_id", "node_id",
    "url", "href", "uri", "link", "api_url",
    "avatar", "icon", "logo", "thumbnail", "image_url", "gravatar",
    "sha", "hash", "etag", "checksum",
    "created_at", "updated_at", "deleted_at", "timestamp",
    "created", "modified", "closed_at", "merged_at", "pushed_at",
    "permissions", "node_id", "events_url", "hooks_url",
];

/// Key suffixes that mark a noisy value.
const NOISY_SUFFIXES: &[&str] = &["_url", "_uri", "_href", "_id", "_at", "_sha"];

/// String prefixes that mark a link-like (noisy) value.
const LINK_PREFIXES: &[&str] = &["http://", "https://", "ssh://", "git@"];

static CAMEL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([a-z])([A-Z])").expect("valid regex"));
static ACRONYM_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([A-Z]+)([A-Z][a-z])").expect("valid regex"));

/// Python `text[:n]` (by code point).
fn char_prefix(text: &str, n: usize) -> &str {
    match text.char_indices().nth(n) {
        Some((idx, _)) => &text[..idx],
        None => text,
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

/// Formats MCP tool results into compact, readable markdown/plain text.
pub struct ResultFormatter {
    pub llm: Arc<dyn BaseLLM>,
}

impl ResultFormatter {
    pub fn new(llm: Arc<dyn BaseLLM>) -> Self {
        Self { llm }
    }

    /// Render a tool's content blocks; JSON text blocks are compacted and
    /// flattened, errors are prefixed, and oversized output is truncated to
    /// 60% of the model's max tokens.
    pub fn format_tool_result(
        &self,
        tool_name: &str,
        content: Option<&[Value]>,
        is_error: bool,
    ) -> String {
        let content = match content {
            Some(c) if !c.is_empty() => c,
            _ => return format!("Tool '{tool_name}' completed with no output."),
        };

        let mut parts: Vec<String> = Vec::with_capacity(content.len());
        for block in content {
            match block {
                Value::Object(map) => {
                    let is_text = match map.get("type") {
                        None => true,
                        Some(Value::String(t)) => t == "text",
                        Some(_) => false,
                    };
                    if is_text {
                        match map.get("text") {
                            None => parts.push(String::new()),
                            Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
                                Ok(data) => {
                                    let data = Self::compact_json(&data);
                                    parts.push(Self::flatten_to_text(&data, ""));
                                }
                                Err(_) => parts.push(text.clone()),
                            },
                            Some(other) => parts.push(py_str(other)),
                        }
                    } else {
                        parts.push(py_repr(block));
                    }
                }
                other => parts.push(py_str(other)),
            }
        }

        let mut result_text = parts.join("\n");

        if is_error {
            return format!("**Error** from `{tool_name}`:\n{result_text}");
        }

        let max_tokens = (self.llm.get_max_tokens() as f64 * 0.6) as usize;
        if self.llm.get_num_tokens(&result_text) > max_tokens {
            let cutoff = max_tokens * 4;
            result_text = format!("{}\n\n... (truncated)", char_prefix(&result_text, cutoff));
        }

        result_text
    }

    /// True for values that carry no meaning for the LLM (ids, urls, hashes,
    /// timestamps, null/bool, ...).
    pub fn is_noisy_value(key: &str, value: &Value) -> bool {
        if matches!(value, Value::Null | Value::Bool(_)) {
            return true;
        }
        let key_lower = key.to_lowercase();
        if NOISY_KEYS.contains(&key_lower.as_str()) {
            return true;
        }
        if NOISY_SUFFIXES.iter().any(|s| key_lower.ends_with(s)) {
            return true;
        }
        if let Value::String(s) = value {
            let stripped = s.trim();
            if LINK_PREFIXES.iter().any(|p| stripped.starts_with(p)) {
                return true;
            }
            if stripped.chars().count() >= 32 && stripped.chars().all(|c| c.is_ascii_hexdigit()) {
                return true;
            }
        }
        false
    }

    /// Recursively drop noisy keys and unwrap single-child wrapper objects.
    pub fn compact_json(data: &Value) -> Value {
        match data {
            Value::Array(items) => Value::Array(items.iter().map(Self::compact_json).collect()),
            Value::Object(map) => {
                let mut compacted = Map::new();
                for (key, value) in map {
                    if Self::is_noisy_value(key, value) {
                        continue;
                    }
                    match value {
                        Value::Object(_) => {
                            let nested = Self::compact_json(value);
                            if truthy(&nested) {
                                compacted.insert(key.clone(), nested);
                            }
                        }
                        Value::Array(_) => {
                            compacted.insert(key.clone(), Self::compact_json(value));
                        }
                        other => {
                            compacted.insert(key.clone(), other.clone());
                        }
                    }
                }
                // Unwrap single-child wrapper dicts (e.g. GraphQL envelopes
                // like {"data": {"issues": {"nodes": [...]}}}).
                if compacted.len() == 1 {
                    let only_value = compacted.values().next().expect("one value");
                    if matches!(only_value, Value::Object(_) | Value::Array(_)) {
                        return Self::compact_json(only_value);
                    }
                }
                Value::Object(compacted)
            }
            other => other.clone(),
        }
    }

    /// Convert camelCase / snake_case keys to readable lowercase words.
    ///
    /// regularMarketPrice  →  regular market price
    /// fifty_two_week_high →  fifty two week high
    pub fn humanize_key(key: &str) -> String {
        // camelCase → insert space before each uppercase run
        let key = CAMEL_RE.replace_all(key, "${1} ${2}");
        let key = ACRONYM_RE.replace_all(&key, "${1} ${2}");
        key.replace('_', " ").to_lowercase()
    }

    /// Convert a compacted value to plain `key: value` lines.
    ///
    /// Produces output like:
    /// ```text
    /// symbol: AAPL
    /// regular market price: 290.64
    /// volume: 25520114
    /// ```
    /// so the LLM never sees raw JSON syntax and can answer naturally.
    pub fn flatten_to_text(data: &Value, prefix: &str) -> String {
        match data {
            Value::Object(map) => {
                let mut lines: Vec<String> = Vec::new();
                for (key, value) in map {
                    let mut label = Self::humanize_key(key);
                    if !prefix.is_empty() {
                        label = format!("{prefix}{label}");
                    }
                    match value {
                        Value::Object(_) | Value::Array(_) => {
                            let nested = Self::flatten_to_text(value, &format!("{label} / "));
                            if !nested.is_empty() {
                                lines.push(nested);
                            }
                        }
                        other => lines.push(format!("{label}: {}", py_str(other))),
                    }
                }
                lines.join("\n")
            }
            Value::Array(items) => {
                let chunks: Vec<String> = items
                    .iter()
                    .map(|item| match item {
                        Value::Object(_) | Value::Array(_) => Self::flatten_to_text(item, prefix),
                        other => format!("{prefix}{}", py_str(other)),
                    })
                    .collect();
                chunks.join("\n")
            }
            other => format!("{prefix}{}", py_str(other)),
        }
    }

    /// Render JSON as a markdown table (arrays of objects) or bullet list.
    pub fn json_to_markdown(data: &Value) -> String {
        match data {
            Value::Array(items) => {
                let rows: Vec<&Map<String, Value>> = items.iter().filter_map(Value::as_object).collect();
                if rows.is_empty() {
                    return items.iter().map(py_str).collect::<Vec<_>>().join("\n");
                }

                let mut columns: Vec<&String> = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for row in &rows {
                    for key in row.keys() {
                        if seen.insert(key) {
                            columns.push(key);
                        }
                    }
                }

                let is_renderable = |val: Option<&Value>| -> bool {
                    match val {
                        Some(Value::Array(_)) => false,
                        Some(Value::Object(obj)) => {
                            obj.contains_key("name") || obj.contains_key("title") || obj.len() == 1
                        }
                        _ => true,
                    }
                };

                let mut scalar_cols: Vec<&String> = columns
                    .iter()
                    .copied()
                    .filter(|col| rows.iter().any(|row| is_renderable(row.get(col.as_str()))))
                    .collect();
                if scalar_cols.is_empty() {
                    scalar_cols = columns.iter().copied().take(5).collect();
                }

                let header = format!(
                    "| {} |",
                    scalar_cols.iter().map(|c| c.as_str()).collect::<Vec<_>>().join(" | ")
                );
                let sep = format!("| {} |", scalar_cols.iter().map(|_| "---").collect::<Vec<_>>().join(" | "));
                let mut lines = vec![header, sep];
                for row in &rows {
                    let mut cells: Vec<String> = Vec::with_capacity(scalar_cols.len());
                    for col in &scalar_cols {
                        let text = match row.get(col.as_str()) {
                            None | Some(Value::Null) => String::new(),
                            Some(Value::Object(val)) => {
                                if let Some(name) = val.get("name") {
                                    py_str(name)
                                } else if let Some(title) = val.get("title") {
                                    py_str(title)
                                } else if val.len() == 1 {
                                    py_str(val.values().next().expect("one value"))
                                } else {
                                    val.iter()
                                        .map(|(k, v)| format!("{k}: {}", py_str(v)))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                }
                            }
                            Some(Value::Array(val)) => {
                                val.iter().take(5).map(py_str).collect::<Vec<_>>().join(", ")
                            }
                            Some(other) => py_str(other),
                        };
                        let mut cell = text.replace('|', "\\|").replace('\n', " ");
                        if cell.chars().count() > 80 {
                            cell = format!("{}...", char_prefix(&cell, 77));
                        }
                        cells.push(cell);
                    }
                    lines.push(format!("| {} |", cells.join(" | ")));
                }
                lines.join("\n")
            }
            Value::Object(map) => {
                let mut lines: Vec<String> = Vec::new();
                for (key, value) in map {
                    match value {
                        Value::Object(_) | Value::Array(_) => {
                            let nested = Self::json_to_markdown(value);
                            lines.push(format!("**{key}**:\n{nested}"));
                        }
                        Value::Null => lines.push(format!("- **{key}**: ")),
                        other => lines.push(format!("- **{key}**: {}", py_str(other))),
                    }
                }
                lines.join("\n")
            }
            other => py_str(other),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn humanizes_keys() {
        assert_eq!(ResultFormatter::humanize_key("regularMarketPrice"), "regular market price");
        assert_eq!(ResultFormatter::humanize_key("fifty_two_week_high"), "fifty two week high");
        assert_eq!(ResultFormatter::humanize_key("HTTPServer"), "http server");
    }

    #[test]
    fn compacts_and_flattens() {
        let data = json!({"data": {"quote": {"symbol": "AAPL", "id": 5, "url": "https://x", "regularMarketPrice": 290.64, "open": 3.0}}});
        let compacted = ResultFormatter::compact_json(&data);
        assert_eq!(compacted, json!({"symbol": "AAPL", "regularMarketPrice": 290.64, "open": 3.0}));
        assert_eq!(
            ResultFormatter::flatten_to_text(&compacted, ""),
            "symbol: AAPL\nregular market price: 290.64\nopen: 3.0"
        );
    }

    #[test]
    fn markdown_table() {
        let data = json!([{"name": "a", "owner": {"name": "bob"}, "tags": [1, 2]}, {"name": "b|c", "flag": true}]);
        let md = ResultFormatter::json_to_markdown(&data);
        assert_eq!(
            md,
            "| name | owner | tags | flag |\n| --- | --- | --- | --- |\n| a | bob | 1, 2 |  |\n| b\\|c |  |  | True |"
        );
    }
}
