//! Port of `smartloop/db/feedback_indexer.py`: vector index for thumbed-up
//! Q&A pairs, so positively-rated conversation exchanges can be retrieved
//! during exploration.

use std::sync::Arc;

use crate::db::store::{ChunkResult, VectorStore};
use crate::db::store_factory::StoreFactory;
use crate::feedback_store::FeedbackEntry;

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
        let _ = (project_dir, base_dir, StoreFactory::create_store);
        todo!()
    }

    /// Build an indexer over an explicit store (test seam).
    pub fn with_store(store: Arc<dyn VectorStore>) -> Self {
        Self { store }
    }

    /// Index one feedback entry as a single `"Q: ...\nA: ..."` chunk.
    pub fn index_feedback(&self, entry: &FeedbackEntry) -> anyhow::Result<()> {
        let _ = (entry, &self.store);
        todo!()
    }

    /// Remove a feedback entry from the index; failures are logged, not raised.
    pub fn remove_feedback(&self, feedback_id: &str) {
        let _ = feedback_id;
        todo!()
    }

    /// Search feedback pairs (Python defaults: `n_results=5`,
    /// `max_distance=0.65`); failures are logged and yield an empty list.
    pub fn search(&self, query: &str, n_results: usize, max_distance: f64) -> Vec<ChunkResult> {
        let _ = (query, n_results, max_distance);
        todo!()
    }
}
