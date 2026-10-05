//! Port of `smartloop/utils/doc_splitter.py`.

use std::sync::Arc;

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

use crate::llms::BaseLLM;
use crate::utils::log_utils::LoggerLike;
use crate::utils::py_compat::{py_json_dumps_with, DumpsOptions};
use crate::workspace::types::Document;

/// Max tokens per chunk — set lower than model limit to allow room for prompt.
pub const MAX_TOKENS: usize = 512;

/// Max rows rendered by [`json_to_table`].
const ROW_CAP: usize = 500;

/// Keys under which a top-level object may hold the row array.
const ROW_KEYS: [&str; 6] = ["rows", "data", "items", "results", "players", "values"];

static TABLE_SEPARATOR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?m)^\|\s*[-:|]+\s*\|$").expect("valid regex"));
static PARAGRAPH_BREAK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{2,}").expect("valid regex"));
static IMAGE_PLACEHOLDER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)<!--\s*image\s*-->").expect("valid regex"));

/// Flatten nested objects into `flat` using dot-prefixed keys.
fn flatten_into(obj: &Map<String, Value>, prefix: &str, flat: &mut Map<String, Value>) {
    for (k, v) in obj {
        let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
        match v {
            Value::Object(inner) => flatten_into(inner, &key, flat),
            other => {
                flat.insert(key, other.clone());
            }
        }
    }
}

/// Render one table cell the way the Python `_cell` helper does.
fn cell(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(b)) => if *b { "true" } else { "false" }.to_string(),
        Some(Value::Number(n)) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(0.0);
            if f.is_finite() && f.fract() == 0.0 {
                if f == 0.0 { "0".to_string() } else { format!("{f:.0}") }
            } else {
                format!("{f:.4}")
            }
        }
        Some(Value::Number(n)) => n.to_string(),
        Some(v @ (Value::Array(_) | Value::Object(_))) => py_json_dumps_with(
            v,
            DumpsOptions { indent: None, item_separator: ",", key_separator: ":", ensure_ascii: false },
        ),
        Some(Value::String(s)) => s.clone(),
    }
}

/// Convert a JSON array-of-objects document into a markdown table.
///
/// Handles a top-level array or an object holding one under a common key
/// (`rows`, `data`, `items`, `results`, `players`, `values`).  Each object
/// becomes a table row; columns are the union of keys, with nested objects
/// flattened using dot-prefixed keys (e.g. a player's
/// `"values": {"epa": 0.3, ...}` becomes `values.epa`).  Floats are rounded
/// to 4 decimals and integral floats shown without the decimal so dense
/// stats stay compact.
///
/// Returns `None` when the text isn't JSON or has no tabular structure, so
/// callers fall back to text splitting / LLM extraction.  Deterministic and
/// context-window-independent — a large data file becomes a table without
/// ever loading it into a model (whose window overflows for big documents,
/// silently killing extraction).
pub fn json_to_table(text: &str) -> Option<String> {
    let data: Value = serde_json::from_str(text).ok()?;

    let mut rows: Option<&Vec<Value>> = match &data {
        Value::Array(items) => Some(items),
        _ => None,
    };
    if let Value::Object(map) = &data {
        for key in ROW_KEYS {
            if let Some(Value::Array(candidate)) = map.get(key) {
                if matches!(candidate.first(), Some(Value::Object(_))) {
                    rows = Some(candidate);
                    break;
                }
            }
        }
    }
    let rows = rows?;

    let mut flat_rows: Vec<Map<String, Value>> = Vec::new();
    for row in rows {
        let Value::Object(obj) = row else { continue };
        let mut flat = Map::new();
        flatten_into(obj, "", &mut flat);
        flat_rows.push(flat);
    }
    if flat_rows.is_empty() {
        return None;
    }

    let total_rows = flat_rows.len();
    let truncated = total_rows > ROW_CAP;
    flat_rows.truncate(ROW_CAP);

    let mut columns: Vec<&String> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for r in &flat_rows {
        for k in r.keys() {
            if seen.insert(k) {
                columns.push(k);
            }
        }
    }

    let header: Vec<&str> = columns.iter().map(|c| c.as_str()).collect();
    let mut lines = vec![format!("| {} |", header.join(" | ")), format!("|{}", "---|".repeat(columns.len()))];
    for r in &flat_rows {
        let cells: Vec<String> = columns.iter().map(|c| cell(r.get(c.as_str()))).collect();
        lines.push(format!("| {} |", cells.join(" | ")));
    }
    if truncated {
        lines.push(format!("\n_({ROW_CAP} of {total_rows} rows shown)_"));
    }
    Some(lines.join("\n"))
}

/// True when the text contains a complete markdown table.
///
/// Detection is the header separator row (e.g. `|---|---|`) — a line of only
/// dashes/colons/pipes/spaces.  Content that is already a table needs no LLM
/// extraction: it's the structured representation a data query wants, and
/// feeding it through the extractor risks a context-window overflow.
pub fn is_markdown_table(text: &str) -> bool {
    TABLE_SEPARATOR_RE.is_match(text)
}

/// Utility for splitting documents into chunks.
pub struct DocumentSplitter {
    pub llm: Arc<dyn BaseLLM>,
    pub logger: Arc<dyn LoggerLike>,
}

impl DocumentSplitter {
    pub fn new(llm: Arc<dyn BaseLLM>, logger: Arc<dyn LoggerLike>) -> Self {
        Self { llm, logger }
    }

    /// Generate text chunks from a document file (read as UTF-8) without
    /// creating a full dataset.
    pub fn split_document(&self, document: &Document) -> std::io::Result<Vec<String>> {
        let document_text = std::fs::read_to_string(&document.path)?;
        Ok(self.split_document_text(&document_text))
    }

    /// Split already-converted markdown text into chunks.
    pub fn split_text(&self, text: &str) -> Vec<String> {
        self.split_document_text(text)
    }

    /// Count tokens using the LLM's tokenizer.
    fn count_tokens(&self, text: &str) -> usize {
        self.llm.get_num_tokens(text)
    }

    /// Split text that exceeds `limit` tokens by paragraphs, then by lines.
    fn split_oversized(&self, text: &str, limit: usize) -> Vec<String> {
        let mut chunks: Vec<String> = Vec::new();
        let mut current: Vec<String> = Vec::new();
        let mut current_count = 0usize;

        for para in PARAGRAPH_BREAK_RE.split(text) {
            let para_count = self.count_tokens(para);
            if !current.is_empty() && current_count + para_count > limit {
                chunks.push(current.join("\n\n"));
                current.clear();
                current_count = 0;
            }

            if para_count > limit {
                // paragraph itself exceeds limit — split by lines
                let mut line_buf: Vec<&str> = Vec::new();
                let mut line_count = 0usize;
                for line in para.split('\n') {
                    let lc = self.count_tokens(line);
                    if !line_buf.is_empty() && line_count + lc > limit {
                        chunks.push(line_buf.join("\n"));
                        line_buf.clear();
                        line_count = 0;
                    }
                    line_buf.push(line);
                    line_count += lc;
                }
                if !line_buf.is_empty() {
                    current.push(line_buf.join("\n"));
                    current_count += line_count;
                }
            } else {
                current.push(para.to_string());
                current_count += para_count;
            }
        }

        if !current.is_empty() {
            chunks.push(current.join("\n\n"));
        }

        chunks
            .iter()
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .map(str::to_string)
            .collect()
    }

    /// Split document text into chunks for vector indexing.
    ///
    /// Indexes the entire document as a single chunk when it fits within the
    /// token budget.  Only splits into smaller pieces when the full text
    /// exceeds [`MAX_TOKENS`], using paragraph boundaries to keep related
    /// content together.
    fn split_document_text(&self, document_text: &str) -> Vec<String> {
        // Strip image placeholders
        let cleaned = IMAGE_PLACEHOLDER_RE.replace_all(document_text, "");
        let cleaned = cleaned.trim();
        if cleaned.is_empty() {
            return Vec::new();
        }

        // Keep as single chunk when it fits
        if self.count_tokens(cleaned) <= MAX_TOKENS {
            return vec![cleaned.to_string()];
        }

        // Otherwise split by paragraphs/lines to stay within budget
        self.split_oversized(cleaned, MAX_TOKENS)
    }
}
