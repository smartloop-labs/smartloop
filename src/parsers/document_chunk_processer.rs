//! Port of `smartloop/parsers/document_chunk_processer.py`.
//!
//! Process DocumentChunks from the vector store into DocumentChunks & ContextItems.

use crate::parsers::base_parser::SharedLLM;
use crate::parsers::context_item::ContextItem;
use crate::parsers::document_chunk::DocumentChunk;

/// Logger name used by the Python module.
const LOGGER: &str = "smartloop.parsers.document_chunk_processer";

/// Process DocumentChunks from the vector store into DocumentChunks & ContextItems.
pub struct DocumentChunkProcessor {
    pub llm: SharedLLM,
    /// Vector-store rows (Python `List[Any]` with DocumentChunk-shaped attributes).
    pub document_chunk_models: Vec<DocumentChunk>,
}

impl DocumentChunkProcessor {
    /// Prints "initializing DocumentChunkProcessor..." and logs each chunk.
    pub fn new(llm: SharedLLM, document_chunk_models: Vec<DocumentChunk>) -> Self {
        println!("initializing DocumentChunkProcessor...");

        tracing::info!(target: LOGGER, "DocumentChunkProcessor: initialized with the following chunks:");
        for chunk in &document_chunk_models {
            tracing::info!(
                target: LOGGER,
                "Chunk document_id: {} -- distance: {}",
                chunk.document_id,
                py_optional_float(chunk.distance)
            );
            tracing::debug!(target: LOGGER, "DOCUMENT CHUNK TEXT: {}", chunk.text);
        }

        Self { llm, document_chunk_models }
    }

    /// Converts all vector-store models into DocumentChunks (each gets a
    /// fresh `id`; `metadata`/`score` take their defaults, as in Python).
    pub fn get_document_chunks(&self) -> Vec<DocumentChunk> {
        tracing::info!(target: LOGGER, "Getting get_document_chunks()");
        self.document_chunk_models
            .iter()
            .map(|model| DocumentChunk {
                document_id: model.document_id.clone(),
                paragraph_id: model.paragraph_id.clone(),
                chunk_id: model.chunk_id.clone(),
                total_chunks: model.total_chunks,
                text: model.text.clone(),
                vector: model.vector.clone(),
                distance: model.distance,
                ..DocumentChunk::default()
            })
            .collect()
    }

    /// Return all DocumentChunks as a list of dicts.
    pub fn get_chunks_as_dicts(&self) -> Vec<serde_json::Value> {
        tracing::info!(target: LOGGER, "Getting chunks as DICT for models");
        self.get_document_chunks().iter().map(DocumentChunk::to_dict).collect()
    }

    /// Combine all DocumentChunks into ContextItems that keep paragraphs contiguous.
    pub fn get_context_items(&self) -> Vec<ContextItem> {
        tracing::info!(target: LOGGER, "Getting get_combined_chunks()");
        let mut results = Vec::with_capacity(self.document_chunk_models.len());

        for chunk in &self.document_chunk_models {
            tracing::debug!(
                target: LOGGER,
                "Chunk document_id: {} -- paragraph_id: {} -- chunk_id: {}",
                chunk.document_id,
                chunk.paragraph_id,
                chunk.chunk_id
            );
            tracing::debug!(target: LOGGER, "Chunk text: {}", chunk.text);

            results.push(ContextItem {
                document_id: chunk.document_id.clone(),
                paragraph_id: chunk.paragraph_id.clone(),
                chunk_id: chunk.chunk_id.clone(),
                text: chunk.text.clone(),
                token_count: self.llm.get_num_tokens(&chunk.text),
            });
        }

        results
    }
}

/// Python `%s` of an `Optional[float]`.
fn py_optional_float(value: Option<f64>) -> String {
    match value {
        Some(v) => crate::parsers::json_parser::py_float_repr(v),
        None => "None".to_string(),
    }
}
