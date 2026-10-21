//! Port of `smartloop/db/store.py`: the `VectorStore` abstraction, the
//! `ChunkResult` search result and the keyword-coverage boost shared by all
//! dense-embedding stores.

use std::collections::HashSet;

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

/// Below this many documents there is too little evidence to call a term
/// common, so every query term counts toward the boost.
pub const MIN_DOCS_FOR_TERM_FREQUENCY: usize = 4;
/// A query term found in more than this share of the documents searched
/// ("the", "how", or a topic every document shares) does not discriminate
/// between them and earns no boost.
pub const MAX_TERM_DOC_SHARE: f64 = 0.5;

/// Distinct query terms (lowercased, longer than two characters).
fn query_terms(query_text: &str) -> HashSet<String> {
    TERM_RE
        .find_iter(query_text)
        .map(|m| m.as_str())
        .filter(|t| t.chars().count() > 2)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Lowercased words of a document's name + content.
fn document_words(name: Option<&str>, content: Option<&str>) -> HashSet<String> {
    let haystack = format!("{} {}", name.unwrap_or("").replace('_', " "), content.unwrap_or("")).to_lowercase();
    TERM_RE.find_iter(&haystack).map(|m| m.as_str().to_string()).collect()
}

/// The query terms that tell the given documents apart: those found in at
/// most [`MAX_TERM_DOC_SHARE`] of them.  With fewer than
/// [`MIN_DOCS_FOR_TERM_FREQUENCY`] documents every query term is kept.
///
/// Without this, filler words a long question is full of ("how", "can",
/// "the", "with") matched nearly every document, each collected the full
/// boost, and a 402-document search returned 213 of them at distance 0.000.
pub fn discriminative_terms<'a>(
    query_text: &str,
    documents: impl IntoIterator<Item = (Option<&'a str>, Option<&'a str>)>,
) -> HashSet<String> {
    let terms = query_terms(query_text);
    let docs: Vec<HashSet<String>> = documents.into_iter().map(|(n, c)| document_words(n, c)).collect();
    if docs.len() < MIN_DOCS_FOR_TERM_FREQUENCY {
        return terms;
    }
    let max_docs = (docs.len() as f64 * MAX_TERM_DOC_SHARE).floor() as usize;
    terms
        .into_iter()
        .filter(|t| docs.iter().filter(|words| words.contains(t)).count() <= max_docs)
        .collect()
}

/// The boost for a document containing some of `terms` (see
/// [`keyword_coverage_boost`]).
pub fn terms_coverage_boost(terms: &HashSet<String>, name: Option<&str>, content: Option<&str>) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let words = document_words(name, content);
    let matched = terms.iter().filter(|t| words.contains(t.as_str())).count();
    MAX_TERM_BOOST.min(PER_TERM_BOOST * matched as f64)
}

/// Return the distance reduction for a document based on how many distinct
/// query keywords appear in its name + content.
///
/// Apply it with [`apply_coverage_boost`].  Clamped to [`MAX_TERM_BOOST`].
pub fn keyword_coverage_boost(query_text: &str, name: Option<&str>, content: Option<&str>) -> f64 {
    terms_coverage_boost(&query_terms(query_text), name, content)
}

/// Shrink a cosine distance by a coverage boost.
///
/// Proportional rather than subtracted: subtracting clamped every boosted
/// document at 0.0, erasing the ranking among them and defeating the
/// relevance cutoff downstream.
pub fn apply_coverage_boost(distance: f64, boost: f64) -> f64 {
    distance * (1.0 - boost)
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

    /// Downcasting hook (Python `isinstance(store, IndexStore)`); `None` for
    /// implementations that don't support it.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}
