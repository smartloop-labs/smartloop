//! Port of `smartloop/utils/doc_splitter.py`.

use std::sync::Arc;

use crate::llms::BaseLLM;
use crate::utils::log_utils::LoggerLike;
use crate::workspace::types::Document;

/// Max tokens per chunk — set lower than model limit to allow room for prompt.
pub const MAX_TOKENS: usize = 512;

/// Max rows rendered by [`json_to_table`].
const ROW_CAP: usize = 500;

/// Convert a JSON array-of-objects document into a markdown table.
///
/// Handles a top-level array or an object holding one under a common key
/// (`rows`, `data`, `items`, `results`, `players`, `values`).  Each object
/// becomes a table row; columns are the union of keys, with nested objects
/// flattened using dot-prefixed keys.  Floats are rounded to 4 decimals and
/// integral floats shown without the decimal.
///
/// Returns `None` when the text isn't JSON or has no tabular structure.
pub fn json_to_table(text: &str) -> Option<String> {
    let _ = (text, ROW_CAP);
    todo!()
}

/// True when the text contains a complete markdown table (a header
/// separator row such as `|---|---|`).
pub fn is_markdown_table(text: &str) -> bool {
    let _ = text;
    todo!()
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

    /// Generate text chunks from a document file (read as UTF-8).
    pub fn split_document(&self, document: &Document) -> std::io::Result<Vec<String>> {
        let _ = document;
        todo!()
    }

    /// Split already-converted markdown text into chunks.
    pub fn split_text(&self, text: &str) -> Vec<String> {
        let _ = text;
        todo!()
    }

    /// Count tokens using the LLM's tokenizer.
    fn count_tokens(&self, text: &str) -> usize {
        let _ = text;
        todo!()
    }

    /// Split text that exceeds `limit` tokens by paragraphs, then by lines.
    fn split_oversized(&self, text: &str, limit: usize) -> Vec<String> {
        let _ = (text, limit);
        todo!()
    }

    /// Split document text into chunks for vector indexing: one chunk when it
    /// fits within [`MAX_TOKENS`], otherwise paragraph/line splitting.
    fn split_document_text(&self, document_text: &str) -> Vec<String> {
        let _ = document_text;
        todo!()
    }
}
