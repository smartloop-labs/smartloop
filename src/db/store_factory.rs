//! Port of `smartloop/db/store_factory.py`: factory to create `VectorStore`
//! instances.

use std::sync::Arc;

use crate::db::sqlite_store::IndexStore;
use crate::db::store::VectorStore;

/// Known store backends (string values match the Python `StoreType`).
pub struct StoreType;

impl StoreType {
    pub const SQLITE: &'static str = "sqlite";
    /// Alias mirroring the Python attribute name `StoreType.SQLite`.
    #[allow(non_upper_case_globals)]
    pub const SQLite: &'static str = "sqlite";
}

/// Return the active backend.
///
/// SQLite is the only backend; kept as a function so callers can still ask.
pub fn resolve_default_store_type() -> &'static str {
    StoreType::SQLITE
}

/// Factory to create instances of `VectorStore` implementations.
pub struct StoreFactory;

impl StoreFactory {
    /// Create a vector store instance (SQLite-backed dense embeddings).
    ///
    /// * `project_dir` — project directory path
    /// * `collection_name` — name of the collection/index (Python default `"documents"`)
    /// * `base_dir` — base directory for model cache and configs
    /// * `store_type` — ignored; retained for call-site compatibility.
    pub fn create_store(
        project_dir: &str,
        collection_name: &str,
        base_dir: Option<&str>,
        store_type: Option<&str>,
    ) -> anyhow::Result<Arc<dyn VectorStore>> {
        let _ = store_type;
        Ok(Arc::new(IndexStore::new(project_dir, collection_name, base_dir)?))
    }
}
