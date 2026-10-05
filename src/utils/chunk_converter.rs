//! Port of `smartloop/utils/chunk_converter.py`.
//!
//! Utility to convert chunk arrays into `DocumentChunk` objects.

use serde_json::{Map, Value};

use crate::parsers::document_chunk::DocumentChunk;

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
    let _ = (chunks, document_id);
    todo!()
}
