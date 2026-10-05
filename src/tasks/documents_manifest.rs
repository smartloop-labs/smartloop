//! Port of `smartloop/tasks/documents_manifest.py`.
//!
//! JSONL manifest that tracks the indexing state of a project's documents folder.
//!
//! Each line in the JSONL file is one document entry:
//!     {"doc_id": "...", "filename": "...", "path": "...", "status": "...", "hash": "..."}
//!
//! The manifest is the source of truth for the indexer watcher:
//! - files in the folder but not in the manifest  → new, enqueue for indexing
//! - files in the manifest but not in the folder  → deleted, remove from store + config

use std::collections::HashSet;
use std::path::PathBuf;

use parking_lot::Mutex;
use serde_json::{Map, Value};

/// Entries keyed by `doc_id`, in file order; each value is the raw JSON object.
pub type ManifestEntries = Map<String, Value>;

/// Read/write a per-project JSONL manifest for the documents inbox folder.
pub struct DocumentsManifest {
    path: PathBuf,
    lock: Mutex<()>,
}

impl DocumentsManifest {
    pub fn new(manifest_path: impl Into<PathBuf>) -> Self {
        Self { path: manifest_path.into(), lock: Mutex::new(()) }
    }

    /// Return all entries keyed by `doc_id`.
    pub fn entries(&self) -> ManifestEntries {
        let _ = &self.lock;
        todo!()
    }

    /// Add or update the entry for `doc_id`.
    pub fn upsert(&self, doc_id: &str, filename: &str, path: &str, status: &str, file_hash: &str) {
        let _ = (doc_id, filename, path, status, file_hash);
        todo!()
    }

    /// Update just the status field of an existing entry.
    pub fn update_status(&self, doc_id: &str, status: &str) {
        let _ = (doc_id, status);
        todo!()
    }

    /// Remove an entry from the manifest.
    pub fn remove(&self, doc_id: &str) {
        let _ = doc_id;
        todo!()
    }

    /// Return the set of all tracked filenames.
    pub fn filenames(&self) -> HashSet<String> {
        todo!()
    }

    /// Return the set of all tracked file paths.
    pub fn paths(&self) -> HashSet<String> {
        todo!()
    }

    /// Read the JSONL file and return entries keyed by doc_id (no lock).
    fn load_raw(&self) -> ManifestEntries {
        let _ = &self.path;
        todo!()
    }

    /// Rewrite the JSONL file from the in-memory map (no lock).
    fn write_raw(&self, entries: &ManifestEntries) {
        let _ = (entries, Self::load_raw);
        todo!()
    }
}
