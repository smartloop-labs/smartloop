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
    let _ = (filename, content, &*AMOUNT_RE, &*DATE_RE);
    todo!()
}
