//! Port of `smartloop/parsers/context_item.py`.

use serde::{Deserialize, Serialize};

/// Object combining the text of all DocumentChunks for a paragraph.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ContextItem {
    pub document_id: String,
    pub chunk_id: String,
    pub paragraph_id: String,
    pub text: String,
    pub token_count: usize,
}
