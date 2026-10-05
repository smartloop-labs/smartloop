//! Port of `smartloop/db/__init__.py`.

pub mod embeddings;
pub mod feedback_indexer;
pub mod llama_embeddings;
pub mod metadata_extractor;
pub mod migrate_chroma;
pub mod sqlite_store;
pub mod store;
pub mod store_factory;
pub mod workspace_indexer;

pub use metadata_extractor::extract_document_metadata;
pub use sqlite_store::IndexStore;
pub use store::ChunkResult;
pub use store_factory::StoreFactory;
