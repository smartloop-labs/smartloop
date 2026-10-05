//! Port of `smartloop/utils/chunk_converter.py`.
//!
//! Utility to convert chunk arrays into `DocumentChunk` objects.

use serde_json::{Map, Value};

use crate::parsers::document_chunk::DocumentChunk;
use crate::utils::py_compat::{py_json_dumps, py_str};

/// A chunk field as text: strings verbatim, `null` as empty, anything else
/// the way Python's `str()` renders it.
fn field_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        other => py_str(other),
    }
}

/// Convert an array of chunk dictionaries (from the chunking agent) into
/// `DocumentChunk` objects that all belong to `document_id`.
///
/// Reads `content` (falling back to `text`), prepends `title` when it is not
/// already part of the text, and stores `document`, `chunk_id`, optional
/// `title` and `token_count` as a JSON metadata string.
pub fn convert_chunks_to_document_chunks(
    chunks: &[Map<String, Value>],
    document_id: &str,
) -> Vec<DocumentChunk> {
    let mut document_chunks = Vec::with_capacity(chunks.len());

    for chunk_data in chunks {
        // Extract content - handle different field names
        let mut text = match chunk_data.get("content").or_else(|| chunk_data.get("text")) {
            Some(value) => field_text(value),
            None => String::new(),
        };

        // Handle title if present
        let title = chunk_data.get("title").map(field_text).unwrap_or_default();
        if !title.is_empty() && !text.contains(&title) {
            text = if !text.is_empty() { format!("{title}\n\n{text}") } else { title.clone() };
        }

        // Create metadata from additional fields
        let chunk_id = uuid::Uuid::new_v4().to_string();
        let mut metadata_fields = Map::new();
        metadata_fields.insert("document".into(), Value::String(text.clone()));
        metadata_fields.insert("chunk_id".into(), Value::String(chunk_id.clone()));

        if let Some(title) = chunk_data.get("title") {
            metadata_fields.insert("title".into(), title.clone());
        }
        if let Some(token_count) = chunk_data.get("token_count") {
            metadata_fields.insert("token_count".into(), token_count.clone());
        }

        let metadata = Some(py_json_dumps(&Value::Object(metadata_fields)));

        let paragraph_id = match chunk_data.get("paragraph_id") {
            Some(value) => py_str(value),
            None => uuid::Uuid::new_v4().to_string(),
        };

        // Create DocumentChunk
        document_chunks.push(DocumentChunk {
            document_id: document_id.to_string(),
            paragraph_id,
            chunk_id,
            total_chunks: chunks.len() as i64,
            text,
            metadata,
            ..DocumentChunk::default()
        });
    }

    document_chunks
}
