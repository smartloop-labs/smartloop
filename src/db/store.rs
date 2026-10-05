//! Port of `smartloop/db/store.py`: the `VectorStore` abstraction, the
//! `ChunkResult` search result and the keyword-coverage boost shared by all
//! dense-embedding stores.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// Keyword-coverage boost shared by all dense-embedding stores.  Dense
// embeddings cluster similar documents together (e.g. all invoices), so a
// query naming a specific entity ("Adam Hart") pulls in look-alikes
// ("Aaron Hawkins").  Rewarding documents by the NUMBER of distinct query
// keywords they contain pushes the more complete match to the top.

/// Distance reduction per distinct query term matched.
pub const PER_TERM_BOOST: f64 = 0.15;
/// Cap so a long query can't zero out the distance.
pub const MAX_TERM_BOOST: f64 = 0.60;
/// Word tokenizer used for the keyword-coverage boost (Python `\w+`).
pub static TERM_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\w+").expect("valid regex"));

/// A chunk handed to [`VectorStore::add_chunks`]: a free-form dict with
/// `text`, `document_id`, and optional `chunk_index`, `document_name`,
/// `source_content` plus any extra scalar metadata keys.
pub type Chunk = Map<String, Value>;

/// Return the distance reduction for a document based on how many distinct
/// query keywords appear in its name + content.
///
/// Subtract the returned value from a cosine distance to boost documents that
/// match more of the query (clamped to [`MAX_TERM_BOOST`]).
pub fn keyword_coverage_boost(query_text: &str, name: Option<&str>, content: Option<&str>) -> f64 {
    let _ = (query_text, name, content);
    todo!()
}

/// A single vector search result with text and source metadata.
///
/// All citation data (document_name, page_number, chunk_index, etc.) is
/// stored in the `metadata` dict — no separate citation field needed.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ChunkResult {
    pub text: String,
    pub document_id: String,
    #[serde(default)]
    pub document_name: Option<String>,
    #[serde(default)]
    pub distance: f64,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub source_content: Option<String>,
    /// This chunk's embedding, as the raw BLOB the index stores.  Carried out
    /// of the store so callers that need to compare the chunk against
    /// something else — citation verification checking it against the
    /// finished answer — can do it without re-encoding text that was already
    /// encoded at index time.  `None` from the TF-IDF fallback path, which has
    /// no vectors.  `blob_to_embedding` decodes it.  Never part of the wire
    /// format.
    #[serde(skip)]
    pub embedding: Option<Vec<u8>>,
}

/// Abstract base for vector stores.
///
/// Implementations are synchronous (SQLite + CPU/GPU embedding) and shared
/// across worker threads as `Arc<dyn VectorStore + Send + Sync>`.
pub trait VectorStore: Send + Sync {
    /// Project directory the store lives under.
    fn project_dir(&self) -> &str;

    /// Name of the collection/index (default `"documents"`).
    fn collection_name(&self) -> &str;

    /// Base directory for model cache and configs.
    fn base_dir(&self) -> Option<&str>;

    /// Add document chunks with `text`, `document_id`, and optional `chunk_index`.
    fn add_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()>;

    /// Semantic search, returns top matching texts.  `None` means "no limit"
    /// for stores that support it (the Python base default was 5).
    fn query(&self, query_text: &str, n_results: Option<usize>) -> anyhow::Result<Vec<String>>;

    /// Semantic search returning structured results with source metadata.
    ///
    /// Default implementation delegates to [`VectorStore::query`] with no
    /// metadata.  Implementations should override for richer results.
    /// Python defaults: `n_results=5`, `max_distance=0.8`, `where=None`.
    fn query_with_metadata(
        &self,
        query_text: &str,
        n_results: Option<usize>,
        max_distance: f64,
        where_: Option<&Map<String, Value>>,
    ) -> anyhow::Result<Vec<ChunkResult>> {
        let _ = (max_distance, where_);
        let texts = self.query(query_text, n_results)?;
        Ok(texts
            .into_iter()
            .map(|t| ChunkResult {
                text: t,
                document_id: String::new(),
                distance: 0.0,
                ..Default::default()
            })
            .collect())
    }

    /// Fetch all chunks for the given document IDs (no semantic query).
    fn get_by_document_ids(&self, document_ids: &[String]) -> anyhow::Result<Vec<ChunkResult>> {
        let _ = document_ids;
        Ok(Vec::new())
    }

    /// Return the number of stored chunks in this collection (default 0).
    fn count(&self) -> anyhow::Result<usize> {
        Ok(0)
    }

    /// Remove all chunks for a document.
    fn delete_document(&self, document_id: &str) -> anyhow::Result<()>;
}
