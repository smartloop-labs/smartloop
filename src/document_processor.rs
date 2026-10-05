//! DocumentProcessor — parses pending documents and stores their chunks.
//!
//! Port of `smartloop/document_processor.py`.  The Python class took
//! duck-typed collections (`projects`, `documents`, `gridfs`, `vector_db`);
//! here they are small traits so any backing store can be plugged in.
//! Documents and projects are free-form JSON objects, as in Python.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::llms::BaseLLM;
use crate::parsers::document_chunk::DocumentChunk;
#[allow(unused_imports)]
use crate::parsers::parser_factory::ParserFactory;
#[allow(unused_imports)]
use crate::utils::chunk_converter::convert_chunks_to_document_chunks;

/// Raised when a document is missing (or its parser rejects it).
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct DocumentMissingError(pub String);

/// Chunk output (`{"chunks": [...]}`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkOutput {
    /// List of document chunks
    pub chunks: Vec<Map<String, Value>>,
}

/// A document summary.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Summary {
    pub id: String,
    pub text: String,
    pub num_of_tokens: i64,
}

/// Document key flagging that the token limit was reached.
pub const IS_TOKEN_LIMIT_REACHED: &str = "is_token_limit_reached";
/// Document key holding the document's token count.
pub const NUM_OF_TOKENS: &str = "num_of_tokens";

/// A free-form document / project record.
pub type Record = Map<String, Value>;

/// Project lookup (`projects.get(project_id)`).
pub trait ProjectRepository: Send + Sync {
    fn get(&self, project_id: &str) -> Option<Record>;
}

/// Document records (`documents.*`).
pub trait DocumentRepository: Send + Sync {
    fn get_all_pending(&self, project_id: &str) -> Vec<Record>;
    fn update(&self, document: &Record);
    fn find_by_project(&self, project_id: &str) -> Vec<Record>;
}

/// Blob store (`gridfs.get(file_id).read()`).
pub trait FileStore: Send + Sync {
    fn get(&self, file_id: &Value) -> anyhow::Result<Vec<u8>>;
}

/// Vector store operations used here.
pub trait ChunkVectorStore: Send + Sync {
    fn open_project(&self, project_id: &str);
    fn delete_all_chunks_by_document_id(&self, document_id: &str);
    fn upsert_document_chunks(&self, chunks: Vec<DocumentChunk>);
}

/// Collects pending documents, parses them and stores them in vector storage.
pub struct DocumentProcessor {
    pub llm: Arc<dyn BaseLLM + Send + Sync>,
    pub projects: Arc<dyn ProjectRepository>,
    pub documents: Arc<dyn DocumentRepository>,
    pub gridfs: Arc<dyn FileStore>,
    pub vector_db: Arc<dyn ChunkVectorStore>,
    pub section_token_size: usize,
}

impl DocumentProcessor {
    /// Build a processor (`section_token_size` defaults to 512 in Python).
    pub fn new(
        llm: Arc<dyn BaseLLM + Send + Sync>,
        projects: Arc<dyn ProjectRepository>,
        documents: Arc<dyn DocumentRepository>,
        gridfs: Arc<dyn FileStore>,
        vector_db: Arc<dyn ChunkVectorStore>,
        section_token_size: usize,
    ) -> Self {
        DocumentProcessor { llm, projects, documents, gridfs, vector_db, section_token_size }
    }

    /// Find all currently pending documents and process each one.
    pub fn process_pending_documents(&self, project_id: &str) {
        let _ = project_id;
        todo!()
    }

    /// Process an individual document (errors are logged, never raised; the
    /// document is always marked not pending and updated).
    pub async fn process_document(&self, document: &mut Record) {
        let _ = document;
        todo!()
    }

    /// Sum of `num_of_tokens` over the project's other documents.
    pub fn count_processed_tokens_for_project(&self, project_id: &str, document_id: &Value) -> i64 {
        let _ = (project_id, document_id);
        todo!()
    }

    /// Split text into sections under `target_tokens_per_section`, by
    /// markdown headers first, then paragraphs, sentences and words.
    pub fn get_sections(&self, text: &str, target_tokens_per_section: usize) -> Vec<String> {
        let _ = (text, target_tokens_per_section);
        todo!()
    }

    fn split_large_section(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let _ = (text, max_tokens);
        todo!()
    }

    fn split_by_sentences(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let _ = (text, max_tokens);
        todo!()
    }

    fn split_by_characters(&self, text: &str, max_tokens: usize) -> Vec<String> {
        let _ = (text, max_tokens);
        todo!()
    }

    /// Break a large document into sections and turn each into a chunk.
    pub fn process_large_document_in_sections(
        &self,
        document_text: &str,
        document_id: &str,
    ) -> Vec<DocumentChunk> {
        let _ = (document_text, document_id);
        todo!()
    }

    /// Parse the document with a type-specific parser, chunk it and upsert
    /// the chunks; records summary/token fields on `document`.
    pub fn process_document_as_chunks(
        &self,
        project: &Record,
        document: &mut Record,
    ) -> anyhow::Result<()> {
        let _ = (project, document);
        todo!()
    }
}
