//! Port of `smartloop/parsers/document_chunk.py`.

use serde::{Deserialize, Serialize};

fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn zero() -> Option<f64> {
    Some(0.0)
}

/// Stores one "chunk" of a paragraph's text up to 500 words based on current
/// config. If there are more words then the paragraph text will flow into
/// multiple chunks all tied back to the `paragraph_id`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentChunk {
    /// All chunks have a unique id.
    #[serde(default = "new_id")]
    pub id: String,
    pub document_id: String,
    pub paragraph_id: String,
    pub chunk_id: String,
    pub total_chunks: i64,
    /// Contains summary and keyword.
    #[serde(default)]
    pub metadata: Option<String>,
    pub text: String,
    #[serde(default = "zero")]
    pub distance: Option<f64>,
    #[serde(default = "zero")]
    pub score: Option<f64>,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
}

impl Default for DocumentChunk {
    fn default() -> Self {
        Self {
            id: new_id(),
            document_id: String::new(),
            paragraph_id: String::new(),
            chunk_id: String::new(),
            total_chunks: 0,
            metadata: None,
            text: String::new(),
            distance: zero(),
            score: zero(),
            vector: None,
        }
    }
}

impl DocumentChunk {
    /// Python `model_dump()` / `to_dict()`: every field, `None` as `null`.
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("DocumentChunk serializes")
    }
}
