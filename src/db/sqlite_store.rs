//! Port of `smartloop/db/sqlite_store.py`: SQLite vector store with dense
//! embeddings and a TF-IDF fallback.
//!
//! Uses the GGUF embedding model (see [`crate::db::embeddings`]) for dense
//! similarity search and falls back to a TF-IDF ranking (a faithful port of
//! scikit-learn's `TfidfVectorizer` + `cosine_similarity`) when the embedding
//! model is unavailable.  All source types (document, feedback, workspace,
//! conversation) share a single `indexes` table differentiated by `type` in
//! `<project_dir>/db/store.db`.  The schema, legacy-table migrations and the
//! `embedding_model.txt` marker are byte-compatible with the Python store.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use ndarray::{Array2, ArrayView1};
use once_cell::sync::Lazy;
use regex::Regex;
use rusqlite::types::{Value as SqlValue, ValueRef};
use rusqlite::{params_from_iter, Connection, Row};
use serde_json::{Map, Value};

use crate::db::embeddings::{
    active_embedding_identity, blob_to_embedding, embedding_to_blob, encode_texts,
    try_cosine_distances,
};
use crate::db::store::{apply_coverage_boost, discriminative_terms, terms_coverage_boost, Chunk, ChunkResult, VectorStore};

/// Logger name of the Python module.
const LOG_TARGET: &str = "smartloop.db.sqlite_store";

/// Chunks per embedding call when indexing; small enough that a query or a
/// citation check waiting for the model gets it within a few seconds.
pub const INDEX_ENCODE_BATCH: usize = 16;

/// [`encode_texts`] over `texts` in [`INDEX_ENCODE_BATCH`]-sized calls, rows
/// in order; `None` if any batch fails (as one call would have).
pub fn encode_in_batches(texts: &[String], base_dir: Option<&str>) -> Option<Array2<f32>> {
    batched(texts, INDEX_ENCODE_BATCH, |batch| encode_texts(batch, base_dir, None))
}

/// `encode` over `texts` in `size`-sized calls, rows concatenated in order.
fn batched(
    texts: &[String],
    size: usize,
    mut encode: impl FnMut(&[String]) -> Option<Array2<f32>>,
) -> Option<Array2<f32>> {
    if texts.len() <= size {
        return encode(texts);
    }
    let mut parts: Vec<Array2<f32>> = Vec::new();
    for batch in texts.chunks(size) {
        let part = encode(batch)?;
        if part.nrows() != batch.len() {
            return None;
        }
        parts.push(part);
    }
    let views: Vec<_> = parts.iter().map(|p| p.view()).collect();
    ndarray::concatenate(ndarray::Axis(0), &views).ok()
}

/// The unified table name.
pub const TABLE: &str = "indexes";

/// Multiplier for the TF-IDF coordination boost: a document matching all
/// distinct query terms gets its similarity scaled by (1 + this), one
/// matching half by (1 + this/2), etc.  Rewards keyword coverage over raw
/// term frequency.
pub const TFIDF_COORD_BOOST: f64 = 1.0;

/// Map legacy collection names to short type values.
pub const TYPE_MAP: &[(&str, &str)] = &[
    ("documents", "document"),
    ("feedback_pairs", "feedback"),
    ("workspace_files", "workspace"),
];

/// Tables from the very first per-collection layout.
pub const LEGACY_COLLECTION_TABLES: &[&str] = &["documents", "feedback_pairs", "workspace_files"];

/// Source content stored per chunk is capped at this many characters.
const MAX_CONTENT_CHARS: usize = 4000;

/// Keys of a chunk dict that are columns, not metadata.
const RESERVED: &[&str] = &["text", "chunk_index", "document_id", "document_name", "source_content"];

/// scikit-learn's default `token_pattern` as configured by the store.
static TFIDF_TOKEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\b\w\w+\b").expect("valid regex"));

/// `max_features` of the TF-IDF fallback.
const TFIDF_MAX_FEATURES: usize = 5000;

/// Map a collection name to its `type` value (unknown names pass through).
pub fn resolve_type(collection_name: &str) -> String {
    TYPE_MAP
        .iter()
        .find(|(coll, _)| *coll == collection_name)
        .map(|(_, t)| t.to_string())
        .unwrap_or_else(|| collection_name.to_string())
}

/// `<project_dir>/db/store.db`, creating the `db` directory.
fn get_db_path(project_dir: &str) -> anyhow::Result<String> {
    let db_dir = Path::new(project_dir).join("db");
    std::fs::create_dir_all(&db_dir)?;
    Ok(db_dir.join("store.db").to_string_lossy().into_owned())
}

/// Open a connection to the store database.
fn get_conn(db_path: &str) -> anyhow::Result<Connection> {
    let conn = Connection::open(db_path)?;
    // Python's sqlite3 waits up to 5 seconds on a locked database.
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    Ok(conn)
}

/// Apply incremental column renames to an existing table.
fn upgrade_indexes_schema(conn: &Connection) -> anyhow::Result<()> {
    let cols: HashSet<String> = {
        let mut stmt = conn.prepare(&format!("PRAGMA table_info({TABLE})"))?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
        rows.collect::<Result<_, _>>()?
    };
    if cols.is_empty() {
        return Ok(());
    }
    if cols.contains("document_id") && !cols.contains("ref") {
        conn.execute(&format!("ALTER TABLE {TABLE} RENAME COLUMN document_id TO ref"), [])?;
    }
    if !cols.contains("embedding") {
        // Column already exists (race or stale PRAGMA cache) is not an error.
        let _ = conn.execute(&format!("ALTER TABLE {TABLE} ADD COLUMN embedding BLOB"), []);
    }
    // Clean up stale indices from a previous iteration that had a project column
    conn.execute("DROP INDEX IF EXISTS idx_indexes_project_type", [])?;
    conn.execute("DROP INDEX IF EXISTS idx_indexes_project_type_ref", [])?;
    Ok(())
}

/// Create the indexes table and indices if they don't exist yet.
///
/// Also migrates data from older table layouts (per-collection tables, the
/// intermediate `chunks` table, or the `source` table) if present.
fn ensure_schema(db_path: &str) -> anyhow::Result<()> {
    let mut conn = get_conn(db_path)?;
    upgrade_indexes_schema(&conn)?;

    conn.execute(
        &format!(
            "
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                ref TEXT NOT NULL,
                name TEXT,
                text TEXT NOT NULL,
                content TEXT,
                metadata TEXT,
                embedding BLOB
            )
        "
        ),
        [],
    )?;
    conn.execute(&format!("CREATE INDEX IF NOT EXISTS idx_indexes_type ON {TABLE}(type)"), [])?;
    conn.execute(
        &format!("CREATE INDEX IF NOT EXISTS idx_indexes_type_ref ON {TABLE}(type, ref)"),
        [],
    )?;

    // The migration's inserts and drops commit together, as the Python
    // connection context manager did.
    let tx = conn.transaction()?;
    migrate_legacy_tables(&tx)?;
    tx.commit()?;
    Ok(())
}

/// Invalidate stored embeddings when the embedding model changes.
///
/// Embeddings from different models live in different vector spaces, so
/// mixing them (or querying old vectors with a new model) produces nonsense.
/// A marker file records which model produced the stored embeddings; if it
/// differs from the configured model, the `embedding` column is cleared so the
/// lazy backfill recomputes every vector with the new model.
fn ensure_embedding_model(db_path: &str, model_name: &str) {
    let marker = Path::new(db_path)
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join("embedding_model.txt");
    let mut previous: Option<String> = None;
    if marker.exists() {
        previous = std::fs::read_to_string(&marker).ok().map(|s| s.trim().to_string());
    }

    if previous.as_deref() == Some(model_name) {
        return;
    }

    let cleared = get_conn(db_path).and_then(|conn| {
        Ok(conn.execute(
            &format!("UPDATE {TABLE} SET embedding = NULL WHERE embedding IS NOT NULL"),
            [],
        )?)
    });
    match cleared {
        Ok(cleared) => {
            if cleared > 0 {
                tracing::info!(
                    target: LOG_TARGET,
                    "[sqlite] Embedding model changed ({} → {}); cleared {} stored embedding(s) for re-embedding.",
                    previous.as_deref().filter(|p| !p.is_empty()).unwrap_or("unknown"),
                    model_name,
                    cleared
                );
            }
        }
        Err(exc) => {
            tracing::warn!(target: LOG_TARGET, "[sqlite] Could not clear stale embeddings: {exc}");
        }
    }

    let _ = std::fs::write(&marker, model_name);
}

/// Move rows from older table layouts into the unified indexes table.
fn migrate_legacy_tables(conn: &Connection) -> anyhow::Result<()> {
    let existing: HashSet<String> = {
        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<Result<_, _>>()?
    };

    // v1: per-collection tables (documents, feedback_pairs, workspace_files)
    for table in LEGACY_COLLECTION_TABLES {
        if !existing.contains(*table) {
            continue;
        }
        let count: i64 = conn.query_row(&format!("SELECT COUNT(*) FROM [{table}]"), [], |r| r.get(0))?;
        if count == 0 {
            conn.execute(&format!("DROP TABLE [{table}]"), [])?;
            continue;
        }
        let type_val = resolve_type(table);
        conn.execute(
            &format!(
                "INSERT OR IGNORE INTO {TABLE} \
                 (id, type, ref, name, text, content, metadata) \
                 SELECT id, ?, document_id, document_name, text, \
                 source_content, metadata FROM [{table}]"
            ),
            [&type_val],
        )?;
        conn.execute(&format!("DROP TABLE [{table}]"), [])?;
    }

    // v2: unified chunks table with collection column
    if existing.contains("chunks") {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        if count == 0 {
            conn.execute("DROP TABLE chunks", [])?;
        } else {
            for (old_coll, new_type) in TYPE_MAP {
                conn.execute(
                    &format!(
                        "INSERT OR IGNORE INTO {TABLE} \
                         (id, type, ref, name, text, content, metadata) \
                         SELECT id, ?, document_id, document_name, text, \
                         source_content, metadata FROM chunks \
                         WHERE collection = ?"
                    ),
                    [new_type, old_coll],
                )?;
            }
            let placeholders = vec!["?"; TYPE_MAP.len()].join(",");
            conn.execute(
                &format!(
                    "INSERT OR IGNORE INTO {TABLE} \
                     (id, type, ref, name, text, content, metadata) \
                     SELECT id, collection, document_id, document_name, text, \
                     source_content, metadata FROM chunks \
                     WHERE collection NOT IN ({placeholders})"
                ),
                params_from_iter(TYPE_MAP.iter().map(|(coll, _)| *coll)),
            )?;
            conn.execute("DROP TABLE chunks", [])?;
        }
    }

    // v3: source table (same schema, just renamed to indexes)
    if existing.contains("source") {
        let count: i64 = conn.query_row("SELECT COUNT(*) FROM source", [], |r| r.get(0))?;
        if count > 0 {
            conn.execute(
                &format!(
                    "INSERT OR IGNORE INTO {TABLE} \
                     (id, type, ref, name, text, content, metadata) \
                     SELECT id, type, \
                     CASE WHEN ref IS NOT NULL THEN ref ELSE document_id END, \
                     name, text, content, metadata \
                     FROM source"
                ),
                [],
            )?;
        }
        conn.execute("DROP TABLE source", [])?;
    }
    Ok(())
}

/// One row of the `indexes` table.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IndexRow {
    pub id: String,
    pub type_: String,
    pub ref_: String,
    pub name: Option<String>,
    pub text: String,
    pub content: Option<String>,
    pub metadata: Option<String>,
    pub embedding: Option<Vec<u8>>,
}

impl IndexRow {
    /// Read a `SELECT *` row by column name (column order differs between
    /// tables created fresh and tables upgraded in place).
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            id: opt_text(row.get_ref("id")?).unwrap_or_default(),
            type_: opt_text(row.get_ref("type")?).unwrap_or_default(),
            ref_: opt_text(row.get_ref("ref")?).unwrap_or_default(),
            name: opt_text(row.get_ref("name")?),
            text: opt_text(row.get_ref("text")?).unwrap_or_default(),
            content: opt_text(row.get_ref("content")?),
            metadata: opt_text(row.get_ref("metadata")?),
            embedding: match row.get_ref("embedding")? {
                ValueRef::Null => None,
                ValueRef::Blob(b) => Some(b.to_vec()),
                ValueRef::Text(t) => Some(t.to_vec()),
                // Not a vector; treated as missing so it is backfilled.
                ValueRef::Integer(_) | ValueRef::Real(_) => None,
            },
        })
    }

    /// The parsed metadata dict plus `document_id` / `document_name`, as the
    /// Python result builders assemble it.
    fn result_metadata(&self) -> Map<String, Value> {
        let mut meta = match self.metadata.as_deref().filter(|m| !m.is_empty()) {
            Some(raw) => match serde_json::from_str::<Value>(raw) {
                Ok(Value::Object(map)) => map,
                _ => Map::new(),
            },
            None => Map::new(),
        };
        meta.insert("document_id".to_string(), Value::String(self.ref_.clone()));
        if let Some(name) = self.name.as_deref().filter(|n| !n.is_empty()) {
            meta.insert("document_name".to_string(), Value::String(name.to_string()));
        }
        meta
    }

    /// Build the search result for this row.
    fn to_result(&self, distance: f64, embedding: Option<Vec<u8>>) -> ChunkResult {
        ChunkResult {
            text: self.text.clone(),
            document_id: self.ref_.clone(),
            document_name: self.name.clone(),
            distance,
            metadata: self.result_metadata(),
            source_content: self.content.clone(),
            embedding,
        }
    }
}

/// A nullable text column, tolerating the other storage classes the way
/// Python's dynamic typing did.
fn opt_text(value: ValueRef<'_>) -> Option<String> {
    match value {
        ValueRef::Null => None,
        ValueRef::Text(t) => Some(String::from_utf8_lossy(t).into_owned()),
        ValueRef::Blob(b) => Some(String::from_utf8_lossy(b).into_owned()),
        ValueRef::Integer(i) => Some(i.to_string()),
        ValueRef::Real(f) => Some(f.to_string()),
    }
}

/// `str(value)` the way Python formats a dict value into an f-string.
fn py_str(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
    }
}

/// A chunk field that must be text (`chunk["text"]`, `chunk["document_id"]`).
fn required_text(chunk: &Chunk, key: &str) -> anyhow::Result<String> {
    match chunk.get(key) {
        Some(Value::String(s)) => Ok(s.clone()),
        Some(other) if !other.is_null() => Ok(py_str(Some(other))),
        _ => Err(anyhow::anyhow!("KeyError: '{key}'")),
    }
}

/// Bind a JSON value as an SQLite parameter (Python sqlite3 adaptation:
/// bools are ints).
fn sql_param(value: &Value) -> anyhow::Result<SqlValue> {
    Ok(match value {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Integer(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlValue::Integer(i)
            } else {
                SqlValue::Real(n.as_f64().unwrap_or(0.0))
            }
        }
        Value::String(s) => SqlValue::Text(s.clone()),
        other => anyhow::bail!("Error binding parameter - type '{}' is not supported", json_type(other)),
    })
}

fn json_type(value: &Value) -> &'static str {
    match value {
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
        _ => "value",
    }
}

/// `json.dumps(obj)` with Python's default formatting (`", "` / `": "`
/// separators, non-ASCII escaped), so the `metadata` column stays
/// byte-identical to what the Python store wrote.
fn py_json_dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    serde::Serialize::serialize(value, &mut ser).expect("serialising a JSON value cannot fail");
    String::from_utf8(out).expect("formatter emits ASCII")
}

/// serde_json formatter reproducing Python's `json.dumps` defaults.
struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_f64<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, value: f64) -> std::io::Result<()> {
        writer.write_all(py_float_repr(value).as_bytes())
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, fragment: &str) -> std::io::Result<()> {
        // ensure_ascii=True: everything outside printable ASCII as \uXXXX
        // (UTF-16 code units, lowercase hex — DEL included).
        for ch in fragment.chars() {
            if (' '..='~').contains(&ch) {
                writer.write_all(&[ch as u8])?;
            } else {
                let mut buf = [0u16; 2];
                for unit in ch.encode_utf16(&mut buf) {
                    write!(writer, "\\u{:04x}", unit)?;
                }
            }
        }
        Ok(())
    }
}

/// Python's `repr(float)` for finite values (`1.0`, `1e-05`, `12345.678`).
fn py_float_repr(value: f64) -> String {
    if value.fract() == 0.0 && value.abs() < 1e16 {
        return format!("{value:.1}");
    }
    let abs = value.abs();
    if abs != 0.0 && !(1e-4..1e16).contains(&abs) {
        // Scientific notation with a signed, at-least-two-digit exponent.
        let s = format!("{value:e}");
        if let Some((mantissa, exp)) = s.split_once('e') {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ("-", d),
                None => ("+", exp),
            };
            return format!("{mantissa}e{sign}{digits:0>2}");
        }
        return s;
    }
    format!("{value}")
}

/// Vector store backed by SQLite with dense-embedding (and TF-IDF fallback)
/// similarity search.
#[derive(Debug, Clone)]
pub struct IndexStore {
    pub project_dir: String,
    pub collection_name: String,
    pub base_dir: Option<String>,
    db_path: String,
    type_: String,
}

impl IndexStore {
    /// Open (creating/migrating as needed) the store for `collection_name`
    /// (Python default `"documents"`) under `project_dir`.
    pub fn new(project_dir: &str, collection_name: &str, base_dir: Option<&str>) -> anyhow::Result<Self> {
        let db_path = get_db_path(project_dir)?;
        let type_ = resolve_type(collection_name);
        ensure_schema(&db_path)?;
        // Clear stored embeddings if the configured embedding model changed
        // (e.g. switching to the multilingual model) so they're recomputed.
        ensure_embedding_model(&db_path, &active_embedding_identity());
        Ok(Self {
            project_dir: project_dir.to_string(),
            collection_name: collection_name.to_string(),
            base_dir: base_dir.map(str::to_string),
            db_path,
            type_,
        })
    }

    /// Path to `store.db`.
    pub fn db_path(&self) -> &str {
        &self.db_path
    }

    /// The `type` value this store reads and writes.
    pub fn type_value(&self) -> &str {
        &self.type_
    }

    /// Full query with an optional `where_document` (`{"$contains": ...}`)
    /// filter.  `n_results=None` returns all matches under `max_distance`.
    pub fn query_with_metadata_filtered(
        &self,
        query_text: &str,
        n_results: Option<usize>,
        max_distance: f64,
        where_: Option<&Map<String, Value>>,
        where_document: Option<&Map<String, Value>>,
    ) -> anyhow::Result<Vec<ChunkResult>> {
        let rows = self.fetch_rows(where_, where_document)?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        // Always use dense embeddings (semantic recall — handles paraphrases
        // and different languages); fall back to TF-IDF only when the
        // embedding model is unavailable.  The keyword-coverage boost in the
        // dense path plus the downstream per-document relevance filter
        // suppress the clustering noise dense search can otherwise introduce.
        let query_vecs = encode_texts(&[query_text.to_string()], self.base_dir.as_deref(), None);
        if let Some(query_vecs) = query_vecs {
            if query_vecs.nrows() > 0 {
                return self.query_by_embedding(
                    &rows,
                    query_vecs.row(0),
                    n_results,
                    max_distance,
                    query_text,
                );
            }
        }
        Ok(self.query_by_tfidf(&rows, query_text, n_results, max_distance))
    }

    /// Rank rows by cosine similarity against dense embeddings, lazily
    /// backfilling (and persisting) missing row embeddings.
    fn query_by_embedding(
        &self,
        rows: &[IndexRow],
        query_vec: ArrayView1<'_, f32>,
        n_results: Option<usize>,
        max_distance: f64,
        query_text: &str,
    ) -> anyhow::Result<Vec<ChunkResult>> {
        // Separate rows with / without precomputed embeddings
        let mut has_emb: Vec<usize> = Vec::new();
        let mut missing_emb: Vec<usize> = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            if row.embedding.is_some() {
                has_emb.push(i);
            } else {
                missing_emb.push(i);
            }
        }

        let mut emb_matrix: Vec<Vec<f32>> = Vec::new();
        let mut emb_rows: Vec<&IndexRow> = Vec::new();

        // Lazy backfill: compute embeddings for rows that lack them
        let backfilled = if missing_emb.is_empty() {
            None
        } else {
            let texts: Vec<String> = missing_emb.iter().map(|&i| rows[i].text.clone()).collect();
            encode_texts(&texts, self.base_dir.as_deref(), None)
        };

        match backfilled {
            Some(new_vecs) => {
                let updates: Vec<(Vec<u8>, String)> = missing_emb
                    .iter()
                    .enumerate()
                    .map(|(j, &idx)| (embedding_to_blob(new_vecs.row(j)), rows[idx].id.clone()))
                    .collect();
                // Persist backfilled embeddings
                if let Err(exc) = self.persist_embeddings(&updates) {
                    tracing::warn!(target: LOG_TARGET, "[sqlite] Backfill persist failed: {exc}");
                }
                // Use freshly computed vectors for ranking, rebuilt in the
                // original row order.
                let vec_by_idx: HashMap<usize, usize> =
                    missing_emb.iter().enumerate().map(|(j, &idx)| (idx, j)).collect();
                let mut all: Vec<usize> = has_emb.iter().chain(missing_emb.iter()).copied().collect();
                all.sort_unstable();
                for idx in all {
                    match vec_by_idx.get(&idx) {
                        Some(&j) => emb_matrix.push(new_vecs.row(j).to_vec()),
                        None => emb_matrix.push(
                            blob_to_embedding(rows[idx].embedding.as_deref().unwrap_or_default()).to_vec(),
                        ),
                    }
                    emb_rows.push(&rows[idx]);
                }
            }
            None => {
                // No missing rows, or they could not be encoded — rank only
                // rows with embeddings.
                for &idx in &has_emb {
                    emb_matrix
                        .push(blob_to_embedding(rows[idx].embedding.as_deref().unwrap_or_default()).to_vec());
                    emb_rows.push(&rows[idx]);
                }
            }
        }

        if emb_matrix.is_empty() {
            return Ok(self.query_by_tfidf(rows, "", n_results, max_distance));
        }

        let dim = emb_matrix[0].len();
        if emb_matrix.iter().any(|v| v.len() != dim) {
            anyhow::bail!("all input arrays must have the same shape");
        }
        let flat: Vec<f32> = emb_matrix.iter().flatten().copied().collect();
        let matrix = Array2::from_shape_vec((emb_matrix.len(), dim), flat)?;
        let mut distances = try_cosine_distances(query_vec, matrix.view())?;

        // Boost documents by the NUMBER of distinct query keywords they
        // contain so the document the user actually named ("Adam Hart") ranks
        // above embedding look-alikes ("Aaron Hawkins").  See
        // store::keyword_coverage_boost.
        // Only terms that tell the documents apart count; see
        // store::discriminative_terms.
        if !query_text.is_empty() {
            fn row_content(row: &IndexRow) -> &str {
                row.content.as_deref().filter(|c| !c.is_empty()).unwrap_or(&row.text)
            }
            let mut seen: HashSet<&str> = HashSet::new();
            let terms = discriminative_terms(
                query_text,
                emb_rows
                    .iter()
                    .filter(|row| seen.insert(row.ref_.as_str()))
                    .map(|row| (row.name.as_deref(), Some(row_content(row)))),
            );
            for (i, row) in emb_rows.iter().enumerate() {
                let boost = terms_coverage_boost(&terms, row.name.as_deref(), Some(row_content(row)));
                if boost != 0.0 {
                    distances[i] = apply_coverage_boost(f64::from(distances[i]), boost) as f32;
                }
            }
        }

        let indices = argsort(distances.iter().map(|d| f64::from(*d)));

        let mut results: Vec<ChunkResult> = Vec::new();
        let limit = n_results.filter(|n| *n > 0).unwrap_or(emb_rows.len());
        for idx in indices {
            if results.len() >= limit {
                break;
            }
            let dist = f64::from(distances[idx]);
            if dist > max_distance {
                continue;
            }
            // Already in hand — ranking just used it.  Passing it on saves
            // whoever needs to compare this chunk again from re-encoding it.
            let blob = embedding_to_blob(ArrayView1::from(&emb_matrix[idx]));
            results.push(emb_rows[idx].to_result(dist, Some(blob)));
        }
        Ok(results)
    }

    /// Write backfilled embeddings in one transaction.
    fn persist_embeddings(&self, updates: &[(Vec<u8>, String)]) -> anyhow::Result<()> {
        let mut conn = get_conn(&self.db_path)?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(&format!("UPDATE {TABLE} SET embedding = ? WHERE id = ?"))?;
            for (blob, id) in updates {
                stmt.execute(rusqlite::params![blob, id])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// Rank rows using TF-IDF cosine similarity (fallback), a faithful port of
    /// scikit-learn's `TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b",
    /// sublinear_tf=True, max_features=5000)` plus the coordination boost.
    fn query_by_tfidf(
        &self,
        rows: &[IndexRow],
        query_text: &str,
        n_results: Option<usize>,
        max_distance: f64,
    ) -> Vec<ChunkResult> {
        let normalized_query = query_text.replace('_', " ");
        let mut corpus: Vec<&str> = rows
            .iter()
            .map(|row| {
                row.content
                    .as_deref()
                    .filter(|c| !c.is_empty())
                    .unwrap_or(row.text.as_str())
            })
            .collect();
        corpus.push(&normalized_query);

        // An empty vocabulary is sklearn's ValueError.
        let Some(tfidf) = TfidfMatrix::fit_transform(&corpus) else {
            return Vec::new();
        };
        let n_docs = rows.len();
        let query_vec = &tfidf.rows[n_docs];
        let mut similarities: Vec<f64> = (0..n_docs).map(|i| tfidf.cosine(n_docs, i)).collect();

        // Coordination boost: reward documents that contain more of the
        // distinct query terms.  Cosine alone can rank a document repeating
        // one common term above one that matches more of the query, so scale
        // each document's score by its query-term coverage.  The query
        // vector's nonzero columns are exactly the in-vocabulary query terms.
        let query_term_cols: Vec<usize> = query_vec.keys().copied().collect();
        if !query_term_cols.is_empty() {
            for (i, sim) in similarities.iter_mut().enumerate() {
                let matched = query_term_cols
                    .iter()
                    .filter(|c| tfidf.rows[i].get(c).is_some_and(|v| *v > 0.0))
                    .count();
                let coverage = matched as f64 / query_term_cols.len() as f64;
                *sim *= 1.0 + TFIDF_COORD_BOOST * coverage;
            }
        }

        let max_sim = similarities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_sim = if similarities.is_empty() { 0.0 } else { max_sim };
        let distances: Vec<f64> = if max_sim > 0.0 {
            similarities.iter().map(|s| 1.0 - (s / max_sim)).collect()
        } else {
            vec![1.0; similarities.len()]
        };
        let indices = argsort(distances.iter().copied());

        let mut results: Vec<ChunkResult> = Vec::new();
        let limit = n_results.filter(|n| *n > 0).unwrap_or(rows.len());
        for idx in indices {
            if results.len() >= limit {
                break;
            }
            let dist = distances[idx];
            if dist > max_distance {
                continue;
            }
            results.push(rows[idx].to_result(dist, None));
        }
        results
    }

    /// Fetch rows from the indexes table for this type, applying Chroma-style
    /// `where` (`$eq`/`$in`/plain) and `where_document` (`$contains`) filters.
    fn fetch_rows(
        &self,
        where_: Option<&Map<String, Value>>,
        where_document: Option<&Map<String, Value>>,
    ) -> anyhow::Result<Vec<IndexRow>> {
        let mut clauses: Vec<String> = vec!["type = ?".to_string()];
        let mut params: Vec<SqlValue> = vec![SqlValue::Text(self.type_.clone())];

        if let Some(where_) = where_ {
            for (key, condition) in where_ {
                if let Value::Object(ops) = condition {
                    for (op, val) in ops {
                        if op == "$eq" {
                            clauses.push(format!("json_extract(metadata, '$.{key}') = ?"));
                            params.push(sql_param(val)?);
                        } else if op == "$in" {
                            let items: Vec<Value> = match val {
                                Value::Array(items) => items.clone(),
                                Value::String(s) => s.chars().map(|c| Value::String(c.to_string())).collect(),
                                other => anyhow::bail!("'{}' object is not iterable", json_type(other)),
                            };
                            let ph = vec!["?"; items.len()].join(",");
                            clauses.push(format!("json_extract(metadata, '$.{key}') IN ({ph})"));
                            for item in &items {
                                params.push(sql_param(item)?);
                            }
                        }
                    }
                } else {
                    clauses.push(format!("json_extract(metadata, '$.{key}') = ?"));
                    params.push(sql_param(condition)?);
                }
            }
        }

        if let Some(where_document) = where_document {
            let contains = match where_document.get("$contains") {
                Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                Some(Value::Number(n)) if n.as_f64() != Some(0.0) => Some(n.to_string()),
                _ => None,
            };
            if let Some(contains) = contains {
                clauses.push("text LIKE ?".to_string());
                params.push(SqlValue::Text(format!("%{contains}%")));
            }
        }

        let sql = format!("SELECT * FROM {TABLE} WHERE {}", clauses.join(" AND "));

        let conn = get_conn(&self.db_path)?;
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(params.iter()), IndexRow::from_row)?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}

/// Indices that sort `values` ascending (stable; NaN last).
fn argsort(values: impl Iterator<Item = f64>) -> Vec<usize> {
    let values: Vec<f64> = values.collect();
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or_else(|| values[a].is_nan().cmp(&values[b].is_nan()))
    });
    indices
}

/// The fitted TF-IDF matrix: one sparse, L2-normalised row per document
/// (`column -> weight`).
struct TfidfMatrix {
    rows: Vec<BTreeMap<usize, f64>>,
}

impl TfidfMatrix {
    /// `TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b", sublinear_tf=True,
    /// max_features=5000).fit_transform(docs)`.  `None` when the vocabulary
    /// is empty (sklearn raises `ValueError`).
    fn fit_transform(docs: &[&str]) -> Option<Self> {
        // Tokenise (lowercase=True is the default preprocessor).
        let counts: Vec<HashMap<String, usize>> = docs
            .iter()
            .map(|doc| {
                let lowered = doc.to_lowercase();
                let mut c: HashMap<String, usize> = HashMap::new();
                for m in TFIDF_TOKEN_RE.find_iter(&lowered) {
                    *c.entry(m.as_str().to_string()).or_insert(0) += 1;
                }
                c
            })
            .collect();

        // Vocabulary sorted alphabetically (sklearn `_sort_features`).
        let mut term_freq: BTreeMap<&str, usize> = BTreeMap::new();
        let mut doc_freq: HashMap<&str, usize> = HashMap::new();
        for c in &counts {
            for (term, n) in c {
                *term_freq.entry(term.as_str()).or_insert(0) += n;
                *doc_freq.entry(term.as_str()).or_insert(0) += 1;
            }
        }
        if term_freq.is_empty() {
            return None;
        }

        // max_features: keep the most frequent terms across the corpus.
        let mut kept: Vec<&str> = term_freq.keys().copied().collect();
        if kept.len() > TFIDF_MAX_FEATURES {
            let mut by_freq = kept.clone();
            by_freq.sort_by(|a, b| term_freq[b].cmp(&term_freq[a]));
            by_freq.truncate(TFIDF_MAX_FEATURES);
            let keep: HashSet<&str> = by_freq.into_iter().collect();
            kept.retain(|t| keep.contains(t));
        }
        let vocab: HashMap<&str, usize> = kept.iter().enumerate().map(|(i, t)| (*t, i)).collect();

        // smooth_idf: idf = ln((1 + n) / (1 + df)) + 1
        let n = docs.len() as f64;
        let rows = counts
            .iter()
            .map(|c| {
                let mut row: BTreeMap<usize, f64> = BTreeMap::new();
                for (term, count) in c {
                    if let Some(&col) = vocab.get(term.as_str()) {
                        let tf = 1.0 + (*count as f64).ln(); // sublinear_tf
                        let df = doc_freq[term.as_str()] as f64;
                        let idf = ((1.0 + n) / (1.0 + df)).ln() + 1.0;
                        row.insert(col, tf * idf);
                    }
                }
                // norm='l2'
                let norm = row.values().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for v in row.values_mut() {
                        *v /= norm;
                    }
                }
                row
            })
            .collect();
        Some(Self { rows })
    }

    /// `cosine_similarity` between rows `a` and `b` (zero rows give 0).
    fn cosine(&self, a: usize, b: usize) -> f64 {
        let (ra, rb) = (&self.rows[a], &self.rows[b]);
        let dot: f64 = ra.iter().filter_map(|(k, v)| rb.get(k).map(|w| v * w)).sum();
        let na = ra.values().map(|v| v * v).sum::<f64>().sqrt();
        let nb = rb.values().map(|v| v * v).sum::<f64>().sqrt();
        if na == 0.0 || nb == 0.0 {
            0.0
        } else {
            dot / (na * nb)
        }
    }
}

impl VectorStore for IndexStore {
    fn project_dir(&self) -> &str {
        &self.project_dir
    }

    fn collection_name(&self) -> &str {
        &self.collection_name
    }

    fn base_dir(&self) -> Option<&str> {
        self.base_dir.as_deref()
    }

    fn add_chunks(&self, chunks: &[Chunk]) -> anyhow::Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        struct RowData {
            id: String,
            ref_: String,
            name: Option<String>,
            text: String,
            content: SqlValue,
            metadata: Option<String>,
        }

        let mut texts: Vec<String> = Vec::with_capacity(chunks.len());
        let mut row_data: Vec<RowData> = Vec::with_capacity(chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            let document_id = required_text(chunk, "document_id")?;
            let chunk_index = match chunk.get("chunk_index") {
                Some(v) => py_str(Some(v)),
                None => i.to_string(),
            };
            let chunk_id = format!("{document_id}_{chunk_index}");
            let mut meta = Map::new();
            for (k, v) in chunk {
                if RESERVED.contains(&k.as_str()) {
                    continue;
                }
                if matches!(v, Value::String(_) | Value::Number(_) | Value::Bool(_)) {
                    meta.insert(k.clone(), v.clone());
                }
            }
            let content = match chunk.get("source_content") {
                None => SqlValue::Text(String::new()),
                Some(Value::String(s)) => {
                    if s.chars().count() > MAX_CONTENT_CHARS {
                        SqlValue::Text(s.chars().take(MAX_CONTENT_CHARS).collect())
                    } else {
                        SqlValue::Text(s.clone())
                    }
                }
                Some(other) => sql_param(other)?,
            };
            let text = required_text(chunk, "text")?;
            let name = match chunk.get("document_name") {
                None | Some(Value::Null) => None,
                Some(Value::String(s)) => Some(s.clone()),
                Some(other) => Some(py_str(Some(other))),
            };

            texts.push(text.clone());
            row_data.push(RowData {
                id: chunk_id,
                ref_: document_id,
                name,
                text,
                content,
                metadata: if meta.is_empty() { None } else { Some(py_json_dumps(&Value::Object(meta))) },
            });
        }

        // Encode the chunk texts in small batches: the embedding model serves
        // one call at a time, and a whole large file in one call (a 191-chunk
        // source file took 36s) held every interactive encode behind it — the
        // answer's "Checking sources" step sat until indexing finished.
        let embeddings = encode_in_batches(&texts, self.base_dir.as_deref());

        let mut conn = get_conn(&self.db_path)?;
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(&format!(
                "INSERT OR REPLACE INTO {TABLE} \
                 (id, type, ref, name, text, content, metadata, embedding) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ))?;
            for (i, rd) in row_data.iter().enumerate() {
                let blob: Option<Vec<u8>> = embeddings
                    .as_ref()
                    .filter(|e| i < e.nrows())
                    .map(|e| embedding_to_blob(e.row(i)));
                stmt.execute(rusqlite::params![
                    rd.id,
                    self.type_,
                    rd.ref_,
                    rd.name,
                    rd.text,
                    rd.content,
                    rd.metadata,
                    blob
                ])?;
            }
        }
        tx.commit()?;
        tracing::info!(target: LOG_TARGET, "Indexed {} chunks into SQLite store", row_data.len());
        Ok(())
    }

    /// Texts of [`VectorStore::query_with_metadata`] with `max_distance=0.8`.
    fn query(&self, query_text: &str, n_results: Option<usize>) -> anyhow::Result<Vec<String>> {
        let results = self.query_with_metadata_filtered(query_text, n_results, 0.8, None, None)?;
        Ok(results.into_iter().map(|r| r.text).collect())
    }

    fn query_with_metadata(
        &self,
        query_text: &str,
        n_results: Option<usize>,
        max_distance: f64,
        where_: Option<&Map<String, Value>>,
    ) -> anyhow::Result<Vec<ChunkResult>> {
        self.query_with_metadata_filtered(query_text, n_results, max_distance, where_, None)
    }

    fn get_by_document_ids(&self, document_ids: &[String]) -> anyhow::Result<Vec<ChunkResult>> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; document_ids.len()].join(",");
        let conn = get_conn(&self.db_path)?;
        let mut stmt = conn.prepare(&format!(
            "SELECT * FROM {TABLE} WHERE type = ? AND ref IN ({placeholders})"
        ))?;
        let params = std::iter::once(self.type_.as_str()).chain(document_ids.iter().map(String::as_str));
        let rows: Vec<IndexRow> = stmt
            .query_map(params_from_iter(params), IndexRow::from_row)?
            .collect::<Result<_, _>>()?;
        Ok(rows.iter().map(|row| row.to_result(0.0, None)).collect())
    }

    fn count(&self) -> anyhow::Result<usize> {
        let conn = get_conn(&self.db_path)?;
        let n: i64 = conn.query_row(
            &format!("SELECT COUNT(*) FROM {TABLE} WHERE type = ?"),
            [&self.type_],
            |r| r.get(0),
        )?;
        Ok(usize::try_from(n).unwrap_or(0))
    }

    fn delete_document(&self, document_id: &str) -> anyhow::Result<()> {
        let conn = get_conn(&self.db_path)?;
        conn.execute(
            &format!("DELETE FROM {TABLE} WHERE type = ? AND ref = ?"),
            [&self.type_, document_id],
        )?;
        tracing::info!(target: LOG_TARGET, "Deleted chunks for document: {document_id}");
        Ok(())
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn py_json_dumps_matches_python_defaults() {
        let v: Value = serde_json::json!({"a": 1, "b": "é", "c": 1.5, "d": true, "e": 2.0});
        // ensure_ascii=True: non-ASCII is written as a \uXXXX escape.
        assert_eq!(py_json_dumps(&v), r#"{"a": 1, "b": "\u00e9", "c": 1.5, "d": true, "e": 2.0}"#);
    }

    #[test]
    fn py_float_repr_matches_python() {
        assert_eq!(py_float_repr(1e-05), "1e-05");
        assert_eq!(py_float_repr(0.0001), "0.0001");
        assert_eq!(py_float_repr(12345.678), "12345.678");
        assert_eq!(py_float_repr(1e16), "1e+16");
        assert_eq!(py_float_repr(3.0), "3.0");
    }

    #[test]
    fn tfidf_ranks_matching_document_first() {
        let m = TfidfMatrix::fit_transform(&["apple banana", "cherry date", "apple"]).unwrap();
        // query is the last doc ("apple"): doc 0 shares a term, doc 1 does not.
        assert!(m.cosine(2, 0) > 0.0);
        assert_eq!(m.cosine(2, 1), 0.0);
        assert!(TfidfMatrix::fit_transform(&["a", "b"]).is_none());
    }

    /// A store over a fresh `store.db`, bypassing the embedding-model marker
    /// (which reads the app settings).
    fn store_in(dir: &Path, collection: &str) -> IndexStore {
        let project_dir = dir.to_string_lossy().into_owned();
        let db_path = get_db_path(&project_dir).unwrap();
        ensure_schema(&db_path).unwrap();
        IndexStore {
            project_dir,
            collection_name: collection.to_string(),
            base_dir: None,
            db_path,
            type_: resolve_type(collection),
        }
    }

    fn chunk(doc: &str, index: i64, text: &str, extra: &[(&str, Value)]) -> Chunk {
        let mut c = Chunk::new();
        c.insert("text".into(), Value::String(text.into()));
        c.insert("document_id".into(), Value::String(doc.into()));
        c.insert("chunk_index".into(), Value::from(index));
        c.insert("document_name".into(), Value::String(format!("{doc}.pdf")));
        for (k, v) in extra {
            c.insert((*k).to_string(), v.clone());
        }
        c
    }

    /// Clears the `encode_texts` override however the test ends.
    struct HookGuard;

    impl Drop for HookGuard {
        fn drop(&mut self) {
            crate::db::embeddings::set_encode_texts_hook(None);
        }
    }

    #[test]
    fn legacy_tables_are_migrated_into_indexes() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = get_db_path(&dir.path().to_string_lossy()).unwrap();
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE documents (id TEXT PRIMARY KEY, document_id TEXT, document_name TEXT, \
                 text TEXT, source_content TEXT, metadata TEXT);
                 INSERT INTO documents VALUES ('d_0', 'd', 'd.pdf', 'hello', 'src', NULL);
                 CREATE TABLE feedback_pairs (id TEXT PRIMARY KEY, document_id TEXT, document_name TEXT, \
                 text TEXT, source_content TEXT, metadata TEXT);",
            )
            .unwrap();
        }
        ensure_schema(&db_path).unwrap();

        let conn = Connection::open(&db_path).unwrap();
        let tables: HashSet<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(tables, HashSet::from(["indexes".to_string()]));
        let (type_, ref_, name): (String, String, String) = conn
            .query_row("SELECT type, ref, name FROM indexes WHERE id = 'd_0'", [], |r| {
                Ok((r.get(0)?, r.get(1)?, r.get(2)?))
            })
            .unwrap();
        assert_eq!((type_.as_str(), ref_.as_str(), name.as_str()), ("document", "d", "d.pdf"));
    }

    #[test]
    #[serial_test::serial(embeddings)]
    fn tfidf_fallback_round_trip() {
        crate::db::embeddings::set_encode_texts_hook(Some(std::sync::Arc::new(|_t, _b, _o| None)));
        let _guard = HookGuard;

        let dir = tempfile::tempdir().unwrap();
        let store = store_in(dir.path(), "documents");
        let other = store_in(dir.path(), "workspace_files");
        store
            .add_chunks(&[
                chunk("inv", 0, "Invoice for Adam Hart total 540", &[("document_type", json_str("invoice"))]),
                chunk("rec", 0, "Recipe for sourdough bread", &[("flag", Value::Bool(true))]),
            ])
            .unwrap();
        other.add_chunks(&[chunk("ws", 0, "Invoice in the workspace", &[])]).unwrap();

        assert_eq!(store.count().unwrap(), 2);
        assert_eq!(other.count().unwrap(), 1);

        let results = store.query_with_metadata("adam hart invoice", None, 0.8, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "inv");
        assert_eq!(results[0].distance, 0.0);
        assert!(results[0].embedding.is_none());
        assert_eq!(results[0].metadata["document_type"], "invoice");
        assert_eq!(results[0].metadata["document_name"], "inv.pdf");

        // Metadata is stored the way Python's json.dumps wrote it.
        let conn = Connection::open(store.db_path()).unwrap();
        let meta: String = conn
            .query_row("SELECT metadata FROM indexes WHERE id = 'rec_0'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(meta, r#"{"flag": true}"#);
        let content: String = conn
            .query_row("SELECT content FROM indexes WHERE id = 'rec_0'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(content, "");

        let mut where_ = Map::new();
        where_.insert("document_type".into(), serde_json::json!({"$eq": "invoice"}));
        let filtered = store.query_with_metadata("bread", None, 1.0, Some(&where_)).unwrap();
        assert_eq!(filtered.iter().map(|r| r.document_id.as_str()).collect::<Vec<_>>(), vec!["inv"]);

        let fetched = store.get_by_document_ids(&["rec".to_string()]).unwrap();
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].text, "Recipe for sourdough bread");

        store.delete_document("inv").unwrap();
        assert_eq!(store.count().unwrap(), 1);
        assert_eq!(other.count().unwrap(), 1);
    }

    fn json_str(s: &str) -> Value {
        Value::String(s.to_string())
    }

    #[test]
    #[serial_test::serial(embeddings)]
    fn embedding_path_ranks_backfills_and_carries_vectors() {
        const VOCAB: [&str; 3] = ["invoice", "recipe", "bread"];
        let encoded: std::sync::Arc<parking_lot::Mutex<Vec<Vec<String>>>> = Default::default();
        let recorder = encoded.clone();
        let encode = move |texts: &[String], _b: Option<&str>, _o: Option<f64>| {
            recorder.lock().push(texts.to_vec());
            let flat: Vec<f32> = texts
                .iter()
                .flat_map(|t| {
                    let lower = t.to_lowercase();
                    VOCAB.iter().map(move |w| lower.matches(w).count() as f32 + 1e-3)
                })
                .collect();
            Some(Array2::from_shape_vec((texts.len(), VOCAB.len()), flat).unwrap())
        };
        let _guard = HookGuard;

        let dir = tempfile::tempdir().unwrap();
        let store = store_in(dir.path(), "documents");
        // Indexed with no embedder: rows land without vectors.
        crate::db::embeddings::set_encode_texts_hook(Some(std::sync::Arc::new(|_t, _b, _o| None)));
        store
            .add_chunks(&[
                chunk("inv", 0, "Invoice invoice", &[]),
                chunk("rec", 0, "Recipe for bread", &[]),
            ])
            .unwrap();
        let conn = Connection::open(store.db_path()).unwrap();
        let missing: i64 = conn
            .query_row("SELECT COUNT(*) FROM indexes WHERE embedding IS NULL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(missing, 2);

        crate::db::embeddings::set_encode_texts_hook(Some(std::sync::Arc::new(encode)));
        let results = store.query_with_metadata("bread recipe", Some(1), 0.8, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].document_id, "rec");
        let blob = results[0].embedding.clone().expect("vector carried out of the store");
        assert_eq!(blob.len(), VOCAB.len() * 4);

        // The query was encoded, then the two rows lacking vectors (backfill),
        // which were persisted.
        assert_eq!(encoded.lock().len(), 2);
        assert_eq!(encoded.lock()[1], vec!["Invoice invoice".to_string(), "Recipe for bread".to_string()]);
        let missing: i64 = conn
            .query_row("SELECT COUNT(*) FROM indexes WHERE embedding IS NULL", [], |r| r.get(0))
            .unwrap();
        assert_eq!(missing, 0);

        // A second query needs only the query encoded.
        store.query_with_metadata("invoice", None, 0.8, None).unwrap();
        assert_eq!(encoded.lock().len(), 3);
    }
}

#[cfg(test)]
mod batched_encode_tests {
    use super::*;

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| i.to_string()).collect()
    }

    fn row_per_text(batch: &[String]) -> Option<Array2<f32>> {
        Some(Array2::from_shape_fn((batch.len(), 2), |(r, c)| {
            if c == 0 { batch[r].parse::<f32>().unwrap() } else { 1.0 }
        }))
    }

    #[test]
    fn splits_into_calls_and_keeps_row_order() {
        let mut sizes = Vec::new();
        let out = batched(&texts(37), 16, |b| {
            sizes.push(b.len());
            row_per_text(b)
        })
        .unwrap();
        assert_eq!(sizes, vec![16, 16, 5]);
        assert_eq!(out.nrows(), 37);
        assert!((0..37).all(|i| out[[i, 0]] == i as f32));
    }

    #[test]
    fn a_failed_batch_fails_the_whole_encode() {
        let mut call = 0;
        let out = batched(&texts(40), 16, |b| {
            call += 1;
            if call == 2 { None } else { row_per_text(b) }
        });
        assert!(out.is_none());
    }
}
