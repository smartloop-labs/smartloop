//! Port of `smartloop/platform/workspace_db.py`.
//!
//! Workspace-level SQLite persistence for cached cloud data.
//!
//! Stores user profile, model catalog, the curated MCP catalog and the latest
//! release per channel locally so the API can serve responses when the cloud
//! platform is unreachable.  Lives at `<home_dir>/<workspace_id>/workspace.db`.

use std::path::{Path, PathBuf};
use std::time::Duration;

use parking_lot::Mutex;
use rusqlite::types::Value as SqlValue;
use rusqlite::{params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior};
use serde::Serialize;
use serde_json::Value;

use crate::platform::client::JsonObject;
use crate::platform::service::now;
use crate::types::ModelKind;

/// Cache tables: copies of cloud data that the next fetch refills, so one can
/// be dropped and recreated freely.  A cache whose shape changes gets a new
/// table name (as `model_catalog_v2` did) rather than a migration.  State
/// that can't be refetched goes in [`MIGRATIONS`] instead.
/// [`drop_outdated_model_catalog`] discards a catalog table written before
/// its current columns.
pub const SCHEMA: &str = "\
CREATE TABLE IF NOT EXISTS workspace (
    id         TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id                  TEXT PRIMARY KEY DEFAULT 'me',
    email               TEXT NOT NULL,
    name                TEXT,
    is_premium          INTEGER NOT NULL DEFAULT 0,
    is_paid             INTEGER NOT NULL DEFAULT 0,
    tokens_processed    INTEGER NOT NULL DEFAULT 0,
    default_temperature REAL NOT NULL DEFAULT 0.3,
    default_memory      INTEGER NOT NULL DEFAULT 1,
    fetched_at          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_catalog_v2 (
    name           TEXT NOT NULL,
    model_id       TEXT,            -- NULL for a remote model
    size           TEXT,
    description    TEXT,
    access_level   TEXT,
    is_accessible  INTEGER NOT NULL DEFAULT 0,
    sha256         TEXT,
    type           TEXT,
    url            TEXT,
    capabilities   TEXT,            -- JSON array of capability names
    context_length INTEGER,
    param_size_b        REAL,
    param_size_estimated INTEGER,
    active_param_size_b REAL,
    attention           TEXT,
    reasoning_budget_tokens INTEGER,
    model_version       INTEGER,
    thinking_default    INTEGER,
    vision_chat_handler TEXT,
    fetched_at     REAL NOT NULL,
    -- One name can be listed twice: a remote row and a local (GGUF) row.
    PRIMARY KEY (name, type)
);

CREATE TABLE IF NOT EXISTS mcp_catalog (
    slug       TEXT PRIMARY KEY,
    entry      TEXT NOT NULL,   -- JSON of the MCPCatalogEntry
    fetched_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS release_cache (
    channel     TEXT PRIMARY KEY,
    release     TEXT,            -- JSON of the latest published ReleaseRead, or NULL
    fetched_at  REAL NOT NULL
);
";

/// SQLite store for workspace-level data.  Thread-safe: writes share one
/// connection behind a mutex ([`WorkspaceDB::write`]) and reads take a
/// read-only connection from a small pool ([`WorkspaceDB::read`]), so under
/// WAL a read never waits on a write.
pub struct WorkspaceDB {
    db_path: PathBuf,
    writer: Mutex<Connection>,
    readers: Mutex<Vec<Connection>>,
}

impl std::fmt::Debug for WorkspaceDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceDB").field("db_path", &self.db_path).finish()
    }
}

/// A SQLite cell as JSON (Python's `sqlite3` row value).
fn sql_to_json(value: SqlValue) -> Value {
    match value {
        SqlValue::Null => Value::Null,
        SqlValue::Integer(i) => Value::from(i),
        SqlValue::Real(f) => serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null),
        SqlValue::Text(s) => Value::String(s),
        SqlValue::Blob(b) => Value::String(String::from_utf8_lossy(&b).into_owned()),
    }
}

/// Python `bool(cell)` for an INTEGER flag column.
fn sql_truthy(value: &SqlValue) -> bool {
    match value {
        SqlValue::Null => false,
        SqlValue::Integer(i) => *i != 0,
        SqlValue::Real(f) => *f != 0.0,
        SqlValue::Text(s) => !s.is_empty(),
        SqlValue::Blob(b) => !b.is_empty(),
    }
}

/// A cell read as an optional float (`row["fetched_at"]`).
fn sql_to_f64(value: SqlValue) -> Option<f64> {
    match value {
        SqlValue::Integer(i) => Some(i as f64),
        SqlValue::Real(f) => Some(f),
        SqlValue::Text(s) => s.trim().parse().ok(),
        SqlValue::Null | SqlValue::Blob(_) => None,
    }
}

/// `int(data.get(key, default))` for a flag field: a bool or a number.
fn int_flag(data: &JsonObject, key: &str, default: bool) -> anyhow::Result<i64> {
    match data.get(key) {
        None => Ok(default as i64),
        Some(Value::Bool(b)) => Ok(*b as i64),
        Some(Value::Number(n)) => n
            .as_i64()
            .or_else(|| n.as_f64().map(|f| f.trunc() as i64))
            .ok_or_else(|| anyhow::anyhow!("invalid value for {key}: {n}")),
        Some(Value::String(s)) => s
            .trim()
            .parse::<i64>()
            .map_err(|_| anyhow::anyhow!("invalid literal for int() for {key}: {s:?}")),
        Some(other) => Err(anyhow::anyhow!("int() argument for {key} must be a number, not {other}")),
    }
}

/// Bind a JSON value the way Python's `sqlite3` binds the equivalent object:
/// `None` → NULL, `bool`/`int` → INTEGER, `float` → REAL, `str` → TEXT.  A
/// list or dict cannot be bound (Python raises `InterfaceError`).
fn to_sql(key: &str, value: &Value) -> anyhow::Result<SqlValue> {
    Ok(match value {
        Value::Null => SqlValue::Null,
        Value::Bool(b) => SqlValue::Integer(*b as i64),
        Value::Number(n) => match n.as_i64() {
            Some(i) => SqlValue::Integer(i),
            None => SqlValue::Real(n.as_f64().unwrap_or(0.0)),
        },
        Value::String(s) => SqlValue::Text(s.clone()),
        Value::Array(_) | Value::Object(_) => {
            anyhow::bail!("Error binding parameter {key}: type is not supported")
        }
    })
}

/// `data[key]` — a required field (Python raises `KeyError`).
fn required(data: &JsonObject, key: &str) -> anyhow::Result<SqlValue> {
    let value = data.get(key).ok_or_else(|| anyhow::anyhow!("KeyError: '{key}'"))?;
    to_sql(key, value)
}

/// `data.get(key)` — a missing key reads as SQL NULL.
fn optional(data: &JsonObject, key: &str) -> anyhow::Result<SqlValue> {
    match data.get(key) {
        Some(value) => to_sql(key, value),
        None => Ok(SqlValue::Null),
    }
}

/// `data.get(key, default)`.
fn or_default(data: &JsonObject, key: &str, default: Value) -> anyhow::Result<SqlValue> {
    to_sql(key, data.get(key).unwrap_or(&default))
}

/// Python truthiness of a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A stored JSON blob that must be an object.
fn parse_object(blob: &str) -> anyhow::Result<JsonObject> {
    match serde_json::from_str::<Value>(blob)? {
        Value::Object(map) => Ok(map),
        other => Err(anyhow::anyhow!("expected a JSON object in the cache, got {other}")),
    }
}

/// `serde_json` formatter reproducing Python's `json.dumps` defaults:
/// `", "` / `": "` separators and `ensure_ascii=True`, so blobs written here
/// match the ones the Python app wrote.
struct PythonJsonFormatter;

impl serde_json::ser::Formatter for PythonJsonFormatter {
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

    fn write_string_fragment<W: ?Sized + std::io::Write>(&mut self, writer: &mut W, fragment: &str) -> std::io::Result<()> {
        for ch in fragment.chars() {
            if ch.is_ascii() {
                writer.write_all(&[ch as u8])?;
            } else {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    write!(writer, "\\u{:04x}", unit)?;
                }
            }
        }
        Ok(())
    }
}

/// `json.dumps(value)` with Python's default formatting.
pub fn python_json_dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonJsonFormatter);
    value.serialize(&mut ser).expect("serializing a JSON value cannot fail");
    String::from_utf8(out).expect("JSON output is UTF-8")
}

/// The cached catalog's table.  It is `model_catalog_v2`, not `model_catalog`,
/// because the workspace DB is shared with older clients (a docent build
/// embedding an earlier smartloop): they fetch one row per name and write it
/// into `model_catalog`, and a newer client reading that would never see the
/// local entry of a model also served in the cloud.  The old table is left
/// to them.
const CATALOG_TABLE: &str = "model_catalog_v2";

/// How long a connection waits on another process's lock before giving up
/// with `SQLITE_BUSY`.
const BUSY_TIMEOUT: Duration = Duration::from_secs(5);

/// Settings every connection needs; SQLite keeps these per connection.
fn configure(conn: &Connection) -> anyhow::Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)?;
    conn.pragma_update(None, "foreign_keys", true)?;
    Ok(())
}

/// Put the file in WAL mode, so readers in any process run alongside one
/// writer.  The mode is stored in the file; a filesystem that can't do WAL
/// keeps its rollback journal, which is slower but still correct.
fn enable_wal(conn: &Connection) -> anyhow::Result<()> {
    let mode: String = conn.query_row("PRAGMA journal_mode = WAL", [], |r| r.get(0))?;
    if !mode.eq_ignore_ascii_case("wal") {
        tracing::warn!(target: "smartloop.platform", "workspace DB stays in {mode} journal mode");
    }
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    Ok(())
}

/// Schema steps for durable state, applied in order by [`apply_migrations`].
/// Step *n* takes `PRAGMA user_version` from *n* to *n + 1*; [`SCHEMA`] is
/// version 0.
///
/// Older builds (docent, an earlier Studio) keep reading and writing the
/// file after a newer build has migrated it, and dev builds share the same
/// home.  So:
///
/// - A step is append-only and never edited once merged.
/// - A step only adds: tables, indexes, and columns that are nullable or
///   have a default.  No drops, renames, type changes or table rebuilds; a
///   breaking change is a new table plus a step that copies into it.
/// - A step runs inside a transaction, so nothing that can't (`VACUUM`,
///   `PRAGMA journal_mode`, `PRAGMA foreign_keys`).
/// - Older builds don't enable foreign keys, so a cascade can't be relied
///   on to clean up after them.
/// - Importing data from outside the DB (a JSON file) is runtime code
///   guarded by a flag column, not a step here.
const MIGRATIONS: &[&str] = &[];

/// Apply the *steps* this file hasn't had yet, in one write transaction, so
/// two processes opening the file together apply each step once.  A file
/// already past the last step (written by a newer client) is left alone.
pub fn apply_migrations(conn: &mut Connection, steps: &[&str]) -> anyhow::Result<()> {
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let version: i64 = tx.query_row("PRAGMA user_version", [], |r| r.get(0))?;
    if version > steps.len() as i64 {
        tracing::info!(
            target: "smartloop.platform",
            "workspace DB is at schema version {version}, newer than this build's {}",
            steps.len()
        );
    }
    for (index, step) in steps.iter().enumerate().skip(version.max(0) as usize) {
        tx.execute_batch(step)?;
        tx.pragma_update(None, "user_version", index as i64 + 1)?;
    }
    tx.commit()?;
    Ok(())
}

/// Read-only connections kept open for reuse; more open while busy and close
/// when done.
const IDLE_READERS: usize = 4;

/// Drop a catalog table from before its `(name, type)` key.  It only caches
/// the cloud catalog, so the next fetch refills it.
fn drop_outdated_model_catalog(conn: &Connection) -> anyhow::Result<()> {
    let columns: Vec<(String, i64)> = conn
        .prepare(&format!("SELECT name, pk FROM pragma_table_info('{CATALOG_TABLE}')"))?
        .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?
        .collect::<Result<_, _>>()?;
    let keyed_by_type = columns.iter().any(|(name, pk)| name == "type" && *pk > 0);
    if !columns.is_empty() && !keyed_by_type {
        conn.execute_batch(&format!("DROP TABLE {CATALOG_TABLE}"))?;
    }
    Ok(())
}

impl WorkspaceDB {
    /// Open (creating parent dirs and schema as needed) the DB at *db_path*.
    pub fn new(db_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut conn = Connection::open(&db_path)?;
        configure(&conn)?;
        enable_wal(&conn)?;
        drop_outdated_model_catalog(&conn)?;
        conn.execute_batch(SCHEMA)?;
        apply_migrations(&mut conn, MIGRATIONS)?;
        Ok(Self { db_path, writer: Mutex::new(conn), readers: Mutex::new(Vec::new()) })
    }

    /// Path of the database file.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Run *f* in a transaction and commit it.
    ///
    /// The transaction opens with `BEGIN IMMEDIATE`, taking the write lock up
    /// front where the busy wait covers it; a deferred transaction that later
    /// needs to write fails at once if another process holds the lock.  An
    /// error from *f* rolls the transaction back.
    pub fn write<T>(&self, f: impl FnOnce(&Transaction) -> anyhow::Result<T>) -> anyhow::Result<T> {
        let mut conn = self.writer.lock();
        let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let value = f(&tx)?;
        tx.commit()?;
        Ok(value)
    }

    /// Run *f* on a read-only connection.  It sees the last committed state,
    /// and under WAL neither waits on nor blocks a writer in any process.
    pub fn read<T>(&self, f: impl FnOnce(&Connection) -> anyhow::Result<T>) -> anyhow::Result<T> {
        let idle = self.readers.lock().pop();
        let conn = match idle {
            Some(conn) => conn,
            None => self.open_reader()?,
        };
        let result = f(&conn);
        let mut readers = self.readers.lock();
        if readers.len() < IDLE_READERS {
            readers.push(conn);
        }
        result
    }

    fn open_reader(&self) -> anyhow::Result<Connection> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX | OpenFlags::SQLITE_OPEN_URI;
        let conn = Connection::open_with_flags(&self.db_path, flags)?;
        configure(&conn)?;
        Ok(conn)
    }

    // -- Users -----------------------------------------------------------

    /// Return the cached user profile (`email, name, is_premium, is_paid,
    /// number_of_tokens_processed, default_temperature, default_memory`), or `None`.
    pub fn get_user(&self) -> anyhow::Result<Option<JsonObject>> {
        let row = self.read(|conn| {
            Ok(conn
                .query_row(
                    "SELECT email, name, is_premium, is_paid, tokens_processed, \
                     default_temperature, default_memory FROM users WHERE id = 'me'",
                    [],
                    |r| {
                        Ok([
                            r.get::<_, SqlValue>(0)?,
                            r.get::<_, SqlValue>(1)?,
                            r.get::<_, SqlValue>(2)?,
                            r.get::<_, SqlValue>(3)?,
                            r.get::<_, SqlValue>(4)?,
                            r.get::<_, SqlValue>(5)?,
                            r.get::<_, SqlValue>(6)?,
                        ])
                    },
                )
                .optional()?)
        })?;
        let Some([email, name, is_premium, is_paid, tokens, temperature, memory]) = row else {
            return Ok(None);
        };
        let mut user = JsonObject::new();
        user.insert("email".into(), sql_to_json(email));
        user.insert("name".into(), sql_to_json(name));
        user.insert("is_premium".into(), Value::Bool(sql_truthy(&is_premium)));
        user.insert("is_paid".into(), Value::Bool(sql_truthy(&is_paid)));
        user.insert("number_of_tokens_processed".into(), sql_to_json(tokens));
        user.insert("default_temperature".into(), sql_to_json(temperature));
        user.insert("default_memory".into(), Value::Bool(sql_truthy(&memory)));
        Ok(Some(user))
    }

    /// Upsert the user profile from a cloud API response (`email` required).
    pub fn save_user(&self, data: &JsonObject) -> anyhow::Result<()> {
        let email = required(data, "email")?;
        let name = optional(data, "name")?;
        let is_premium = int_flag(data, "is_premium", false)?;
        let is_paid = int_flag(data, "is_paid", false)?;
        let tokens = or_default(data, "number_of_tokens_processed", Value::from(0))?;
        let temperature = or_default(data, "default_temperature", Value::from(0.3))?;
        let memory = int_flag(data, "default_memory", true)?;
        let fetched_at = now();

        self.write(|tx| {
            tx.execute(
                "INSERT INTO users
                       (id, email, name, is_premium, is_paid, tokens_processed,
                        default_temperature, default_memory, fetched_at)
                   VALUES ('me', ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       email = excluded.email,
                       name = excluded.name,
                       is_premium = excluded.is_premium,
                       is_paid = excluded.is_paid,
                       tokens_processed = excluded.tokens_processed,
                       default_temperature = excluded.default_temperature,
                       default_memory = excluded.default_memory,
                       fetched_at = excluded.fetched_at
                ",
                params![email, name, is_premium, is_paid, tokens, temperature, memory, fetched_at],
            )?;
            Ok(())
        })
    }

    /// Timestamp of the last successful cloud fetch, or `None`.
    pub fn user_fetched_at(&self) -> anyhow::Result<Option<f64>> {
        let ts = self.read(|conn| {
            Ok(conn
                .query_row("SELECT fetched_at FROM users WHERE id = 'me'", [], |r| r.get::<_, SqlValue>(0))
                .optional()?)
        })?;
        Ok(ts.and_then(sql_to_f64))
    }

    // -- Models ----------------------------------------------------------

    /// Return the cached model catalog.
    pub fn list_models(&self) -> anyhow::Result<Vec<JsonObject>> {
        self.read(|conn| {
            let mut stmt = conn.prepare(
                "SELECT model_id, name, size, description, access_level, is_accessible, sha256, \
                        type, url, capabilities, context_length, \
                        param_size_b, model_version, thinking_default, vision_chat_handler, \
                        param_size_estimated, active_param_size_b, attention, reasoning_budget_tokens \
                 FROM model_catalog_v2 ORDER BY rowid",
            )?;
            let rows = stmt.query_map([], |r| {
                let mut m = JsonObject::new();
                m.insert("model_id".into(), sql_to_json(r.get(0)?));
                m.insert("name".into(), sql_to_json(r.get(1)?));
                m.insert("size".into(), sql_to_json(r.get(2)?));
                m.insert("description".into(), sql_to_json(r.get(3)?));
                m.insert("access_level".into(), sql_to_json(r.get(4)?));
                m.insert("is_accessible".into(), Value::Bool(sql_truthy(&r.get::<_, SqlValue>(5)?)));
                m.insert("sha256".into(), sql_to_json(r.get(6)?));
                m.insert("type".into(), sql_to_json(r.get(7)?));
                m.insert("url".into(), sql_to_json(r.get(8)?));
                if let SqlValue::Text(caps) = r.get::<_, SqlValue>(9)? {
                    if let Ok(caps @ Value::Array(_)) = serde_json::from_str::<Value>(&caps) {
                        m.insert("capabilities".into(), caps);
                    }
                }
                m.insert("context_length".into(), sql_to_json(r.get(10)?));
                m.insert("param_size_b".into(), sql_to_json(r.get(11)?));
                m.insert("model_version".into(), sql_to_json(r.get(12)?));
                let thinking_default = match r.get::<_, SqlValue>(13)? {
                    SqlValue::Null => Value::Null,
                    flag => Value::Bool(sql_truthy(&flag)),
                };
                m.insert("thinking_default".into(), thinking_default);
                m.insert("vision_chat_handler".into(), sql_to_json(r.get(14)?));
                let param_size_estimated = match r.get::<_, SqlValue>(15)? {
                    SqlValue::Null => Value::Null,
                    flag => Value::Bool(sql_truthy(&flag)),
                };
                m.insert("param_size_estimated".into(), param_size_estimated);
                m.insert("active_param_size_b".into(), sql_to_json(r.get(16)?));
                m.insert("attention".into(), sql_to_json(r.get(17)?));
                m.insert("reasoning_budget_tokens".into(), sql_to_json(r.get(18)?));
                Ok(m)
            })?;
            Ok(rows.collect::<Result<Vec<_>, _>>()?)
        })
    }

    /// Replace the entire model catalog from a cloud API response (`name`
    /// required).
    ///
    /// Every row is validated before anything is written, and the replace is
    /// one transaction, so a malformed entry leaves the previous catalog intact.
    pub fn save_models(&self, models: &[JsonObject]) -> anyhow::Result<()> {
        let fetched_at = now();
        let rows = models
            .iter()
            .map(|m| {
                let capabilities = match m.get("capabilities") {
                    Some(caps @ Value::Array(_)) => SqlValue::Text(caps.to_string()),
                    _ => SqlValue::Null,
                };
                Ok((
                    (
                        optional(m, "model_id")?,
                        required(m, "name")?,
                        optional(m, "size")?,
                        optional(m, "description")?,
                        optional(m, "access_level")?,
                        int_flag(m, "is_accessible", false)?,
                        optional(m, "sha256")?,
                    ),
                    (
                        // Part of the key: a row without a type is a local model's.
                        match optional(m, "type")? {
                            SqlValue::Null => SqlValue::Text(ModelKind::Local.value().to_string()),
                            kind => kind,
                        },
                        optional(m, "url")?,
                        capabilities,
                        optional(m, "context_length")?,
                    ),
                    (
                        optional(m, "param_size_b")?,
                        optional(m, "model_version")?,
                        optional(m, "thinking_default")?,
                        optional(m, "vision_chat_handler")?,
                        optional(m, "param_size_estimated")?,
                        optional(m, "active_param_size_b")?,
                        optional(m, "attention")?,
                        optional(m, "reasoning_budget_tokens")?,
                    ),
                ))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        self.write(|tx| {
            tx.execute("DELETE FROM model_catalog_v2", [])?;
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO model_catalog_v2
                           (model_id, name, size, description, access_level, is_accessible, sha256,
                            type, url, capabilities, context_length,
                            param_size_b, model_version, thinking_default, vision_chat_handler,
                            param_size_estimated, active_param_size_b, attention,
                            reasoning_budget_tokens, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ",
                )?;
                for (
                    (model_id, name, size, description, access_level, is_accessible, sha256),
                    (kind, url, capabilities, context_length),
                    (
                        param_size_b,
                        model_version,
                        thinking_default,
                        vision_chat_handler,
                        param_size_estimated,
                        active_param_size_b,
                        attention,
                        reasoning_budget_tokens,
                    ),
                ) in &rows
                {
                    stmt.execute(params![
                        model_id,
                        name,
                        size,
                        description,
                        access_level,
                        is_accessible,
                        sha256,
                        kind,
                        url,
                        capabilities,
                        context_length,
                        param_size_b,
                        model_version,
                        thinking_default,
                        vision_chat_handler,
                        param_size_estimated,
                        active_param_size_b,
                        attention,
                        reasoning_budget_tokens,
                        fetched_at
                    ])?;
                }
            }
            Ok(())
        })
    }

    /// Timestamp of the last model catalog fetch (`MIN(fetched_at)`), or `None`.
    pub fn models_fetched_at(&self) -> anyhow::Result<Option<f64>> {
        self.min_fetched_at("SELECT MIN(fetched_at) AS ts FROM model_catalog_v2")
    }

    fn min_fetched_at(&self, sql: &str) -> anyhow::Result<Option<f64>> {
        let ts = self.read(|conn| Ok(conn.query_row(sql, [], |r| r.get::<_, SqlValue>(0)).optional()?))?;
        Ok(ts.and_then(sql_to_f64))
    }

    // -- MCP catalog -----------------------------------------------------

    /// Return the cached curated MCP catalog, ordered as the cloud sent it.
    pub fn list_mcp_catalog(&self) -> anyhow::Result<Vec<JsonObject>> {
        let blobs = self.read(|conn| {
            let mut stmt = conn.prepare("SELECT entry FROM mcp_catalog ORDER BY rowid")?;
            let blobs = stmt
                .query_map([], |r| r.get::<_, String>(0))?
                .collect::<Result<Vec<_>, _>>()?;
            Ok(blobs)
        })?;
        blobs.iter().map(|b| parse_object(b)).collect()
    }

    /// Replace the whole cached catalog from a cloud API response; entries
    /// without a `slug` are skipped.
    ///
    /// Entries are stored as JSON blobs rather than columns: they nest
    /// `remote_config` / `local_config` and carry a `tags` list, none of which
    /// flatten usefully, and we only ever read them back whole.
    pub fn save_mcp_catalog(&self, entries: &[JsonObject]) -> anyhow::Result<()> {
        let fetched_at = now();
        let rows = entries
            .iter()
            .filter(|e| e.get("slug").map(truthy).unwrap_or(false))
            .map(|e| Ok((required(e, "slug")?, python_json_dumps(&Value::Object(e.clone())))))
            .collect::<anyhow::Result<Vec<(SqlValue, String)>>>()?;

        self.write(|tx| {
            tx.execute("DELETE FROM mcp_catalog", [])?;
            {
                let mut stmt = tx.prepare(
                    "INSERT INTO mcp_catalog (slug, entry, fetched_at)
                       VALUES (?, ?, ?)
                    ",
                )?;
                for (slug, entry) in &rows {
                    stmt.execute(params![slug, entry, fetched_at])?;
                }
            }
            Ok(())
        })
    }

    /// Timestamp of the last catalog fetch, or `None`.
    pub fn mcp_catalog_fetched_at(&self) -> anyhow::Result<Option<f64>> {
        self.min_fetched_at("SELECT MIN(fetched_at) AS ts FROM mcp_catalog")
    }

    // -- Releases --------------------------------------------------------

    /// Cached latest release for *channel* (Python default `"stable"`), or
    /// `None`.
    ///
    /// `None` is returned both when nothing has been cached and when the
    /// cached value records that the channel has no published release.  To
    /// tell those two states apart, use [`Self::release_fetched_at`] — it
    /// returns a timestamp once the channel has been polled, even if the
    /// release is `None`.
    pub fn get_release(&self, channel: &str) -> anyhow::Result<Option<JsonObject>> {
        let blob = self.read(|conn| {
            Ok(conn
                .query_row(
                    "SELECT release FROM release_cache WHERE channel = ?",
                    params![channel],
                    |r| r.get::<_, Option<String>>(0),
                )
                .optional()?)
        })?;
        match blob.flatten() {
            None => Ok(None),
            Some(blob) => parse_object(&blob).map(Some),
        }
    }

    /// Upsert the latest release for *channel* from a cloud API response.
    ///
    /// *release* may be `None` to record that the channel currently has no
    /// published release (still bumps `fetched_at`).
    pub fn save_release(&self, channel: &str, release: Option<&JsonObject>) -> anyhow::Result<()> {
        let blob = release.map(|r| python_json_dumps(&Value::Object(r.clone())));
        let fetched_at = now();
        self.write(|tx| {
            tx.execute(
                "INSERT INTO release_cache (channel, release, fetched_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(channel) DO UPDATE SET
                       release = excluded.release,
                       fetched_at = excluded.fetched_at
                ",
                params![channel, blob, fetched_at],
            )?;
            Ok(())
        })
    }

    /// Timestamp of the last release fetch for *channel*, or `None`.
    pub fn release_fetched_at(&self, channel: &str) -> anyhow::Result<Option<f64>> {
        let ts = self.read(|conn| {
            Ok(conn
                .query_row(
                    "SELECT fetched_at FROM release_cache WHERE channel = ?",
                    params![channel],
                    |r| r.get::<_, SqlValue>(0),
                )
                .optional()?)
        })?;
        Ok(ts.and_then(sql_to_f64))
    }

    // -- Workspace -------------------------------------------------------

    /// Record this workspace's id.  Inserts on first call; bumps `updated_at`
    /// on subsequent calls.
    pub fn save_workspace(&self, workspace_id: &str) -> anyhow::Result<()> {
        let ts = now();
        self.write(|tx| {
            tx.execute(
                "INSERT INTO workspace (id, created_at, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       updated_at = excluded.updated_at
                ",
                params![workspace_id, ts, ts],
            )?;
            Ok(())
        })
    }

    /// The workspace row `{id, created_at, updated_at}`, or `None` if no
    /// workspace has been recorded.
    pub fn get_workspace(&self) -> anyhow::Result<Option<JsonObject>> {
        let row = self.read(|conn| {
            Ok(conn
                .query_row("SELECT id, created_at, updated_at FROM workspace LIMIT 1", [], |r| {
                    Ok((r.get::<_, SqlValue>(0)?, r.get::<_, SqlValue>(1)?, r.get::<_, SqlValue>(2)?))
                })
                .optional()?)
        })?;
        Ok(row.map(|(id, created_at, updated_at)| {
            let mut ws = JsonObject::new();
            ws.insert("id".into(), sql_to_json(id));
            ws.insert("created_at".into(), sql_to_json(created_at));
            ws.insert("updated_at".into(), sql_to_json(updated_at));
            ws
        }))
    }
}
