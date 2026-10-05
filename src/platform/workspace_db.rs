//! Port of `smartloop/platform/workspace_db.py`.
//!
//! Workspace-level SQLite persistence for cached cloud data.
//!
//! Stores user profile, model catalog, the curated MCP catalog and the latest
//! release per channel locally so the API can serve responses when the cloud
//! platform is unreachable.  Lives at `<home_dir>/<workspace_id>/workspace.db`.

use std::path::{Path, PathBuf};

use parking_lot::Mutex;
use rusqlite::Connection;

use crate::platform::client::JsonObject;

/// Schema (identical to the Python `_SCHEMA`, so existing DBs keep working).
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

CREATE TABLE IF NOT EXISTS model_catalog (
    model_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    size          TEXT,
    description   TEXT,
    access_level  TEXT,
    is_accessible INTEGER NOT NULL DEFAULT 0,
    sha256        TEXT,
    fetched_at    REAL NOT NULL
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

/// Simple SQLite store for workspace-level cached data.  Thread-safe: the
/// connection is guarded by a mutex (Python `check_same_thread=False` + lock).
pub struct WorkspaceDB {
    db_path: PathBuf,
    conn: Mutex<Connection>,
}

impl std::fmt::Debug for WorkspaceDB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkspaceDB").field("db_path", &self.db_path).finish()
    }
}

impl WorkspaceDB {
    /// Open (creating parent dirs and schema as needed) the DB at *db_path*.
    pub fn new(db_path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let _ = db_path;
        todo!()
    }

    /// Path of the database file.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    // -- Users -----------------------------------------------------------

    /// Return the cached user profile (`email, name, is_premium, is_paid,
    /// number_of_tokens_processed, default_temperature, default_memory`), or `None`.
    pub fn get_user(&self) -> anyhow::Result<Option<JsonObject>> {
        let _ = &self.conn;
        todo!()
    }

    /// Upsert the user profile from a cloud API response (`email` required).
    pub fn save_user(&self, data: &JsonObject) -> anyhow::Result<()> {
        let _ = data;
        todo!()
    }

    /// Timestamp of the last successful cloud fetch, or `None`.
    pub fn user_fetched_at(&self) -> anyhow::Result<Option<f64>> {
        todo!()
    }

    // -- Models ----------------------------------------------------------

    /// Return the cached model catalog.
    pub fn list_models(&self) -> anyhow::Result<Vec<JsonObject>> {
        todo!()
    }

    /// Replace the entire model catalog (`model_id`, `name` required).
    pub fn save_models(&self, models: &[JsonObject]) -> anyhow::Result<()> {
        let _ = models;
        todo!()
    }

    /// Timestamp of the last model catalog fetch (`MIN(fetched_at)`), or `None`.
    pub fn models_fetched_at(&self) -> anyhow::Result<Option<f64>> {
        todo!()
    }

    // -- MCP catalog -----------------------------------------------------

    /// Return the cached curated MCP catalog, ordered as the cloud sent it.
    pub fn list_mcp_catalog(&self) -> anyhow::Result<Vec<JsonObject>> {
        todo!()
    }

    /// Replace the whole cached catalog; entries without a `slug` are skipped.
    /// Entries are stored as JSON blobs and read back whole.
    pub fn save_mcp_catalog(&self, entries: &[JsonObject]) -> anyhow::Result<()> {
        let _ = entries;
        todo!()
    }

    /// Timestamp of the last catalog fetch, or `None`.
    pub fn mcp_catalog_fetched_at(&self) -> anyhow::Result<Option<f64>> {
        todo!()
    }

    // -- Releases --------------------------------------------------------

    /// Cached latest release for *channel*, or `None` (both when nothing has
    /// been cached and when the channel is known to have no release; use
    /// [`Self::release_fetched_at`] to tell them apart).
    pub fn get_release(&self, channel: &str) -> anyhow::Result<Option<JsonObject>> {
        let _ = channel;
        todo!()
    }

    /// Upsert the latest release for *channel*; `None` records "no published
    /// release" and still bumps `fetched_at`.
    pub fn save_release(&self, channel: &str, release: Option<&JsonObject>) -> anyhow::Result<()> {
        let _ = (channel, release);
        todo!()
    }

    /// Timestamp of the last release fetch for *channel*, or `None`.
    pub fn release_fetched_at(&self, channel: &str) -> anyhow::Result<Option<f64>> {
        let _ = channel;
        todo!()
    }

    // -- Workspace -------------------------------------------------------

    /// Record this workspace's id. Inserts on first call; bumps `updated_at` after.
    pub fn save_workspace(&self, workspace_id: &str) -> anyhow::Result<()> {
        let _ = workspace_id;
        todo!()
    }

    /// The workspace row `{id, created_at, updated_at}`, or `None`.
    pub fn get_workspace(&self) -> anyhow::Result<Option<JsonObject>> {
        todo!()
    }
}
