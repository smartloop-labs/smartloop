//! Port of `smartloop/db/metadata_extractor.py`: lightweight metadata
//! extraction — no LLM required.
//!
//! Parses the document filename with simple string/regex rules to produce a
//! flat dict of metadata stored alongside each chunk.  The vector search
//! itself handles entity-based queries by semantic similarity; the metadata is
//! only used for cheap pre-filtering when an exact attribute match is possible
//! (e.g. `document_type=invoice`).

use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{Map, Value};

/// Filename words recognised as a document type.
pub const DOC_TYPE_KEYWORDS: &[&str] = &[
    "invoice", "receipt", "contract", "agreement", "report", "statement", "order", "quote",
    "proposal", "letter", "memo", "summary", "plan",
];

static AMOUNT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(\d[\d,]*(?:\.\d+)?)").expect("valid regex"));
static SEPARATOR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[_\-]+").expect("valid regex"));
static DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(\d{4}[-_]\d{2}[-_]\d{2}|\d{2}[-_]\d{2}[-_]\d{4}|\d{4})").expect("valid regex")
});

/// Return a flat metadata dict parsed from the filename.
///
/// No LLM is used — extraction is pure string/regex so it adds zero latency
/// to the indexing pipeline.  (`content` and the Python `llm` argument are
/// accepted for call-site parity and unused.)
///
/// Detected fields (all optional): `document_type`, `date` (YYYY-MM-DD or
/// YYYY), `amount` (first numeric value; int or float).
pub fn extract_document_metadata(filename: &str, content: &str) -> Map<String, Value> {
    let _ = content;
    let stem = path_stem(filename);
    // Normalise separators to spaces
    let normalised = SEPARATOR_RE.replace_all(&stem, " ");
    let parts: Vec<&str> = normalised.split_whitespace().collect();

    let mut meta = Map::new();

    // document_type
    for part in &parts {
        let lowered = part.to_lowercase();
        if DOC_TYPE_KEYWORDS.contains(&lowered.as_str()) {
            meta.insert("document_type".to_string(), Value::String(lowered));
            break;
        }
    }

    // date
    if let Some(caps) = DATE_RE.captures(&stem) {
        meta.insert("date".to_string(), Value::String(caps[1].replace('_', "-")));
    }

    // amount (first standalone number that looks like a currency value)
    if let Some(caps) = AMOUNT_RE.captures(&stem) {
        let val = caps[1].replace(',', "");
        if val.contains('.') {
            if let Ok(f) = val.parse::<f64>() {
                if let Some(n) = serde_json::Number::from_f64(f) {
                    meta.insert("amount".to_string(), Value::Number(n));
                }
            }
        } else if let Ok(i) = val.parse::<i64>() {
            meta.insert("amount".to_string(), Value::Number(i.into()));
        } else if let Ok(u) = val.parse::<u64>() {
            meta.insert("amount".to_string(), Value::Number(u.into()));
        } else if let Ok(f) = val.parse::<f64>() {
            // Python ints are unbounded; past u64 keep the value as a float
            // rather than dropping it.
            if let Some(n) = serde_json::Number::from_f64(f) {
                meta.insert("amount".to_string(), Value::Number(n));
            }
        }
    }

    meta
}

/// `pathlib.Path(filename).stem`: the final component without its last
/// suffix (a leading dot does not start a suffix).
fn path_stem(filename: &str) -> String {
    let trimmed = filename.trim_end_matches('/');
    let name = trimmed.rsplit('/').next().unwrap_or(trimmed);
    match name.rfind('.') {
        Some(idx) if idx > 0 && idx < name.len() - 1 => name[..idx].to_string(),
        _ => name.to_string(),
    }
}
