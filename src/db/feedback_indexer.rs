//! Port of `smartloop/db/feedback_indexer.py`: vector index for thumbed-up
//! Q&A pairs, so positively-rated conversation exchanges can be retrieved
//! during exploration.

use std::sync::Arc;

use serde_json::Value;

use crate::db::store::{Chunk, ChunkResult, VectorStore};
use crate::db::store_factory::StoreFactory;
use crate::feedback_store::FeedbackEntry;

/// Logger name of the Python module.
const LOG_TARGET: &str = "smartloop.db.feedback_indexer";

/// Collection (index `type`) holding feedback pairs.
pub const FEEDBACK_COLLECTION: &str = "feedback_pairs";

/// Indexes and searches thumbed-up feedback pairs.
pub struct FeedbackIndexer {
    store: Arc<dyn VectorStore>,
}

impl FeedbackIndexer {
    /// Create an indexer backed by the project's store
    /// (`StoreFactory::create_store(project_dir, FEEDBACK_COLLECTION, base_dir)`).
    pub fn new(project_dir: &str, base_dir: Option<&str>) -> anyhow::Result<Self> {
        let store = StoreFactory::create_store(project_dir, FEEDBACK_COLLECTION, base_dir, None)?;
        Ok(Self { store })
    }

    /// Build an indexer over an explicit store (test seam).
    pub fn with_store(store: Arc<dyn VectorStore>) -> Self {
        Self { store }
    }

    /// Index one feedback entry as a single `"Q: ...\nA: ..."` chunk.
    pub fn index_feedback(&self, entry: &FeedbackEntry) -> anyhow::Result<()> {
        let text = format!("Q: {}\nA: {}", entry.question, entry.answer);
        let mut chunk = Chunk::new();
        chunk.insert("text".to_string(), Value::String(text));
        chunk.insert("document_id".to_string(), Value::String(entry.feedback_id.clone()));
        chunk.insert("chunk_index".to_string(), Value::from(0));
        chunk.insert(
            "document_name".to_string(),
            Value::String(format!("feedback/{}/{}", entry.session_id, entry.message_id)),
        );
        self.store.add_chunks(&[chunk])?;
        tracing::info!(target: LOG_TARGET, "[feedback-index] Indexed {}", entry.feedback_id);
        Ok(())
    }

    /// Remove a feedback entry from the index; failures are logged, not raised.
    pub fn remove_feedback(&self, feedback_id: &str) {
        match self.store.delete_document(feedback_id) {
            Ok(()) => tracing::info!(target: LOG_TARGET, "[feedback-index] Removed {feedback_id}"),
            Err(e) => tracing::warn!(target: LOG_TARGET, "[feedback-index] Remove failed: {e}"),
        }
    }

    /// Search feedback pairs (Python defaults: `n_results=5`,
    /// `max_distance=0.65`); failures are logged and yield an empty list.
    pub fn search(&self, query: &str, n_results: usize, max_distance: f64) -> Vec<ChunkResult> {
        match self.store.query_with_metadata(query, Some(n_results), max_distance, None) {
            Ok(results) => results,
            Err(e) => {
                tracing::warn!(target: LOG_TARGET, "[feedback-index] Search failed: {e}");
                Vec::new()
            }
        }
    }
}
