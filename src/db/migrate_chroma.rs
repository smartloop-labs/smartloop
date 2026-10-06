//! Port of `smartloop/db/migrate_chroma.py`: one-time migration from ChromaDB
//! (`chroma.sqlite3`) to `IndexStore` (`store.db`).
//!
//! Reads collections and their documents directly from the ChromaDB SQLite
//! file and inserts them into the `IndexStore`.  After a successful migration
//! `chroma.sqlite3` is renamed to `chroma.sqlite3.bak` so it is not repeated.

use std::collections::HashMap;
use std::path::Path;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags};
use serde_json::Value;

use crate::db::sqlite_store::IndexStore;
use crate::db::store::{Chunk, VectorStore};

/// Logger name of the Python module.
const LOG_TARGET: &str = "smartloop.db.migrate_chroma";

/// Migrate data from ChromaDB to IndexStore for a single project.
///
/// Returns `true` if migration was performed, `false` if skipped (no chroma db
/// found or already migrated) or if it failed (failure is logged).
pub fn migrate_chroma_to_index_store(project_dir: &str) -> bool {
    let chroma_path = Path::new(project_dir).join("db").join("chroma.sqlite3");
    if !chroma_path.exists() {
        return false;
    }
    let chroma_path = chroma_path.to_string_lossy().into_owned();

    let result = (|| -> anyhow::Result<()> {
        let collections = read_chroma_collections(&chroma_path)?;
        if collections.is_empty() {
            archive_chroma(&chroma_path);
            return Ok(());
        }

        for (collection_name, chunks) in &collections {
            if chunks.is_empty() {
                continue;
            }
            let store = IndexStore::new(project_dir, collection_name, None)?;
            store.add_chunks(chunks)?;
            tracing::info!(
                target: LOG_TARGET,
                "[migrate] Migrated {} chunks from ChromaDB collection '{}'",
                chunks.len(),
                collection_name
            );
        }

        archive_chroma(&chroma_path);
        tracing::info!(target: LOG_TARGET, "[migrate] ChromaDB migration complete for {project_dir}");
        Ok(())
    })();

    match result {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(target: LOG_TARGET, "[migrate] ChromaDB migration failed for {project_dir}: {e}");
            false
        }
    }
}

/// Rename `chroma.sqlite3` → `chroma.sqlite3.bak` (logs on failure).
fn archive_chroma(chroma_path: &str) {
    let bak = format!("{chroma_path}.bak");
    if let Err(e) = std::fs::rename(chroma_path, &bak) {
        tracing::warn!(target: LOG_TARGET, "[migrate] Could not rename {chroma_path}: {e}");
    }
}

/// Read all collections and their documents from a ChromaDB SQLite file.
///
/// Returns a map of collection name → chunk dicts suitable for
/// `IndexStore::add_chunks()`.  Collection order follows first appearance in
/// the `embeddings` table, like the Python dict.
fn read_chroma_collections(chroma_path: &str) -> anyhow::Result<Vec<(String, Vec<Chunk>)>> {
    let conn = Connection::open_with_flags(
        chroma_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE | OpenFlags::SQLITE_OPEN_URI,
    )?;
    let mut result: Vec<(String, Vec<Chunk>)> = Vec::new();

    // Map collection id → name
    let coll_map: HashMap<String, String> = {
        let mut stmt = conn.prepare("SELECT id, name FROM collections")?;
        let rows = stmt.query_map([], |row| Ok((text_of(row.get_ref(0)?), text_of(row.get_ref(1)?))))?;
        rows.collect::<Result<_, _>>()?
    };

    // Map metadata segment id → collection id
    let seg_to_coll: HashMap<String, String> = {
        let mut stmt = conn.prepare("SELECT id, collection FROM segments WHERE scope = 'METADATA'")?;
        let rows = stmt.query_map([], |row| Ok((text_of(row.get_ref(0)?), text_of(row.get_ref(1)?))))?;
        rows.collect::<Result<_, _>>()?
    };

    // Read all embeddings grouped by segment
    let emb_rows: Vec<(i64, String, String)> = {
        let mut stmt = conn.prepare("SELECT id, segment_id, embedding_id FROM embeddings")?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, text_of(row.get_ref(1)?), text_of(row.get_ref(2)?)))
        })?;
        rows.collect::<Result<_, _>>()?
    };

    // Group embedding internal ids by collection name, in first-seen order
    let mut emb_by_coll: Vec<(String, Vec<(i64, String)>)> = Vec::new();
    for (internal_id, segment_id, embedding_id) in emb_rows {
        let Some(coll_id) = seg_to_coll.get(&segment_id) else {
            continue;
        };
        let Some(coll_name) = coll_map.get(coll_id) else {
            continue;
        };
        match emb_by_coll.iter_mut().find(|(name, _)| name == coll_name) {
            Some((_, list)) => list.push((internal_id, embedding_id)),
            None => emb_by_coll.push((coll_name.clone(), vec![(internal_id, embedding_id)])),
        }
    }

    // For each collection, read metadata for all embeddings
    let mut stmt = conn.prepare(
        "SELECT key, string_value, int_value, float_value FROM embedding_metadata WHERE id = ?",
    )?;
    for (coll_name, embeddings) in emb_by_coll {
        let mut chunks = Vec::new();
        for (internal_id, embedding_id) in embeddings {
            let meta_rows: Vec<ChromaMetaRow> = stmt
                .query_map([internal_id], |row| {
                    Ok(ChromaMetaRow {
                        key: text_of(row.get_ref(0)?),
                        string_value: opt_text(row.get_ref(1)?),
                        int_value: row.get::<_, Option<i64>>(2).unwrap_or(None),
                        float_value: row.get::<_, Option<f64>>(3).unwrap_or(None),
                    })
                })?
                .collect::<Result<_, _>>()?;
            if let Some(chunk) = meta_rows_to_chunk(&meta_rows, &embedding_id) {
                chunks.push(chunk);
            }
        }
        result.push((coll_name, chunks));
    }

    Ok(result)
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
pub fn meta_rows_to_chunk(meta_rows: &[ChromaMetaRow], embedding_id: &str) -> Option<Chunk> {
    let _ = embedding_id;
    let mut text: Option<String> = None;
    let mut document_id = String::new();
    let mut document_name: Option<String> = None;
    let mut source_content: Option<String> = None;
    let mut chunk_index: i64 = 0;
    let mut extra: Vec<(String, Value)> = Vec::new();

    for row in meta_rows {
        match row.key.as_str() {
            "chroma:document" => text = row.string_value.clone(),
            "document_id" => document_id = row.string_value.clone().unwrap_or_default(),
            "document_name" => document_name = row.string_value.clone(),
            "source_content" => source_content = row.string_value.clone(),
            "chunk_index" => chunk_index = row.int_value.unwrap_or(0),
            key => {
                // Python: `string_value or int_value or float_value` — the
                // first truthy value, else the last one if it is not None.
                let val: Option<Value> = if row.string_value.as_deref().is_some_and(|s| !s.is_empty()) {
                    row.string_value.clone().map(Value::String)
                } else if row.int_value.is_some_and(|i| i != 0) {
                    row.int_value.map(Value::from)
                } else {
                    row.float_value.and_then(serde_json::Number::from_f64).map(Value::Number)
                };
                if let Some(val) = val {
                    match extra.iter_mut().find(|(k, _)| k == key) {
                        Some(entry) => entry.1 = val,
                        None => extra.push((key.to_string(), val)),
                    }
                }
            }
        }
    }

    let text = text?;

    let mut chunk = Chunk::new();
    chunk.insert("text".to_string(), Value::String(text));
    chunk.insert("document_id".to_string(), Value::String(document_id));
    chunk.insert("document_name".to_string(), document_name.map(Value::String).unwrap_or(Value::Null));
    chunk.insert("chunk_index".to_string(), Value::from(chunk_index));
    chunk.insert("source_content".to_string(), Value::String(source_content.unwrap_or_default()));
    for (key, val) in extra {
        chunk.insert(key, val);
    }
    Some(chunk)
}

/// A text column read as a string (non-text values formatted).
fn text_of(value: ValueRef<'_>) -> String {
    opt_text(value).unwrap_or_default()
}

/// A nullable text column.
fn opt_text(value: ValueRef<'_>) -> Option<String> {
    match value {
        ValueRef::Null => None,
        ValueRef::Text(t) | ValueRef::Blob(t) => Some(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Integer(i) => Some(i.to_string()),
        ValueRef::Real(f) => Some(f.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(key: &str, s: Option<&str>, i: Option<i64>, f: Option<f64>) -> ChromaMetaRow {
        ChromaMetaRow { key: key.into(), string_value: s.map(str::to_string), int_value: i, float_value: f }
    }

    #[test]
    fn meta_rows_become_a_chunk() {
        let chunk = meta_rows_to_chunk(
            &[
                row("chroma:document", Some("hello"), None, None),
                row("document_id", Some("d1"), None, None),
                row("chunk_index", None, Some(3), None),
                row("page_number", None, Some(2), None),
                row("empty", Some(""), Some(0), None),
                row("score", None, None, Some(0.5)),
            ],
            "e1",
        )
        .unwrap();
        assert_eq!(chunk["text"], "hello");
        assert_eq!(chunk["document_id"], "d1");
        assert_eq!(chunk["document_name"], Value::Null);
        assert_eq!(chunk["chunk_index"], 3);
        assert_eq!(chunk["source_content"], "");
        assert_eq!(chunk["page_number"], 2);
        assert_eq!(chunk["score"], 0.5);
        assert!(!chunk.contains_key("empty"));
    }

    #[test]
    fn rows_without_document_text_are_skipped() {
        assert!(meta_rows_to_chunk(&[row("document_id", Some("d1"), None, None)], "e1").is_none());
    }
}
