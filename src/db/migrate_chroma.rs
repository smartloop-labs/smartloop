//! Port of `smartloop/db/migrate_chroma.py`: one-time migration from ChromaDB
//! (`chroma.sqlite3`) to `IndexStore` (`store.db`).
//!
//! Reads collections and their documents directly from the ChromaDB SQLite
//! file and inserts them into the `IndexStore`.  After a successful migration
//! `chroma.sqlite3` is renamed to `chroma.sqlite3.bak` so it is not repeated.

use std::collections::HashMap;

use crate::db::sqlite_store::IndexStore;
use crate::db::store::Chunk;

/// Migrate data from ChromaDB to IndexStore for a single project.
///
/// Returns `true` if migration was performed, `false` if skipped (no chroma db
/// found or already migrated) or if it failed (failure is logged).
pub fn migrate_chroma_to_index_store(project_dir: &str) -> bool {
    let _ = (project_dir, IndexStore::new);
    todo!()
}

/// Rename `chroma.sqlite3` → `chroma.sqlite3.bak` (logs on failure).
fn archive_chroma(chroma_path: &str) {
    let _ = chroma_path;
    todo!()
}

/// Read all collections and their documents from a ChromaDB SQLite file.
///
/// Returns a map of collection name → chunk dicts suitable for
/// `IndexStore::add_chunks()`.  Collection order follows first appearance in
/// the `embeddings` table, like the Python dict.
fn read_chroma_collections(chroma_path: &str) -> anyhow::Result<Vec<(String, Vec<Chunk>)>> {
    let _ = (chroma_path, HashMap::<String, String>::new);
    todo!()
}

/// One row of `embedding_metadata`: `(key, string_value, int_value, float_value)`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChromaMetaRow {
    pub key: String,
    pub string_value: Option<String>,
    pub int_value: Option<i64>,
    pub float_value: Option<f64>,
}

/// Convert ChromaDB metadata rows into a chunk dict for IndexStore, or `None`
/// when the rows carry no `chroma:document` text.
fn meta_rows_to_chunk(meta_rows: &[ChromaMetaRow], embedding_id: &str) -> Option<Chunk> {
    let _ = (meta_rows, embedding_id);
    todo!()
}
