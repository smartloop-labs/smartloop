//! Port of `smartloop/parsers/document_chunk_processer.py`.
//!
//! Process DocumentChunks from the vector store into DocumentChunks & ContextItems.

use crate::parsers::base_parser::SharedLLM;
use crate::parsers::context_item::ContextItem;
use crate::parsers::document_chunk::DocumentChunk;

/// Process DocumentChunks from the vector store into DocumentChunks & ContextItems.
pub struct DocumentChunkProcessor {
    pub llm: SharedLLM,
    /// Vector-store rows (Python `List[Any]` with DocumentChunk-shaped attributes).
    pub document_chunk_models: Vec<DocumentChunk>,
}

impl DocumentChunkProcessor {
    /// Prints "initializing DocumentChunkProcessor..." and logs each chunk.
    pub fn new(llm: SharedLLM, document_chunk_models: Vec<DocumentChunk>) -> Self {
        let _ = (llm, document_chunk_models);
        todo!()
    }

    /// Converts all vector-store models into DocumentChunks.
    pub fn get_document_chunks(&self) -> Vec<DocumentChunk> {
        todo!()
    }

    /// Return all DocumentChunks as a list of dicts.
    pub fn get_chunks_as_dicts(&self) -> Vec<serde_json::Value> {
        todo!()
    }

    /// Combine all DocumentChunks into ContextItems that keep paragraphs contiguous.
    pub fn get_context_items(&self) -> Vec<ContextItem> {
        todo!()
    }
}
