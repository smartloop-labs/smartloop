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
use std::io;
use std::path::PathBuf;

use parking_lot::Mutex;
use serde::Serialize;
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

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /// Return all entries keyed by `doc_id`.
    pub fn entries(&self) -> ManifestEntries {
        if !self.path.exists() {
            return ManifestEntries::new();
        }
        let _guard = self.lock.lock();
        self.load_raw()
    }

    /// Add or update the entry for `doc_id`.
    pub fn upsert(&self, doc_id: &str, filename: &str, path: &str, status: &str, file_hash: &str) {
        let mut entry = Map::new();
        entry.insert("doc_id".to_string(), Value::String(doc_id.to_string()));
        entry.insert("filename".to_string(), Value::String(filename.to_string()));
        entry.insert("path".to_string(), Value::String(path.to_string()));
        entry.insert("status".to_string(), Value::String(status.to_string()));
        entry.insert("hash".to_string(), Value::String(file_hash.to_string()));

        let _guard = self.lock.lock();
        let mut current = self.load_raw();
        current.insert(doc_id.to_string(), Value::Object(entry));
        self.write_raw(&current);
    }

    /// Update just the status field of an existing entry.
    pub fn update_status(&self, doc_id: &str, status: &str) {
        let _guard = self.lock.lock();
        let mut current = self.load_raw();
        if let Some(entry) = current.get_mut(doc_id) {
            if let Some(obj) = entry.as_object_mut() {
                obj.insert("status".to_string(), Value::String(status.to_string()));
            }
            self.write_raw(&current);
        }
    }

    /// Remove an entry from the manifest.
    pub fn remove(&self, doc_id: &str) {
        let _guard = self.lock.lock();
        let mut current = self.load_raw();
        if current.shift_remove(doc_id).is_some() {
            self.write_raw(&current);
        }
    }

    /// Return the set of all tracked filenames.
    pub fn filenames(&self) -> HashSet<String> {
        self.field_values("filename")
    }

    /// Return the set of all tracked file paths.
    pub fn paths(&self) -> HashSet<String> {
        self.field_values("path")
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    fn field_values(&self, field: &str) -> HashSet<String> {
        self.entries()
            .values()
            .filter_map(|e| e.get(field))
            .map(|v| match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .collect()
    }

    /// Read the JSONL file and return entries keyed by doc_id (no lock).
    ///
    /// Lines that are not JSON or lack a `doc_id` are skipped.  A later
    /// line for the same `doc_id` replaces the earlier one in place, as a
    /// Python dict assignment does.
    fn load_raw(&self) -> ManifestEntries {
        let mut result = ManifestEntries::new();
        if !self.path.exists() {
            return result;
        }
        let Ok(bytes) = std::fs::read(&self.path) else {
            return result;
        };
        let text = String::from_utf8_lossy(&bytes);
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(entry) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            let key = match entry.get("doc_id") {
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => continue,
            };
            result.insert(key, entry);
        }
        result
    }

    /// Rewrite the JSONL file from the in-memory map (no lock).
    fn write_raw(&self, entries: &ManifestEntries) {
        if let Err(exc) = self.try_write_raw(entries) {
            tracing::warn!(
                target: "smartloop.tasks.documents_manifest",
                "Could not write manifest {}: {exc}",
                self.path.display()
            );
        }
    }

    fn try_write_raw(&self, entries: &ManifestEntries) -> io::Result<()> {
        if let Some(parent) = self.path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let lines: Vec<String> = entries.values().map(python_dumps).collect();
        let mut body = lines.join("\n");
        if !lines.is_empty() {
            body.push('\n');
        }
        std::fs::write(&self.path, body)
    }
}

/// `json.dumps(value, ensure_ascii=False)`: Python's default `", "` / `": "`
/// separators, non-ASCII written as-is.
fn python_dumps(value: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonFormatter);
    if value.serialize(&mut ser).is_err() {
        return value.to_string();
    }
    String::from_utf8(out).unwrap_or_else(|_| value.to_string())
}

/// serde_json formatter emitting Python's default separators.
struct PythonFormatter;

impl serde_json::ser::Formatter for PythonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: ?Sized + io::Write>(&mut self, writer: &mut W, first: bool) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}
