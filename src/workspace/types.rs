//! Port of `smartloop/workspace/types.py`.
//!
//! Lightweight workspace types shared between the workspace and model layers.
//!
//! Kept in a separate module so the model layer can use these without pulling
//! in the full workspace module.

use serde::{Deserialize, Serialize};

/// Render `path` relative to the project, else to the framework home.
///
/// Tried in order:
///
/// * relative to `project_dir` — `documents/invoice.pdf`, the form that
///   survives the whole framework being copied to another machine, user, or
///   sandbox container;
/// * relative to `AppSettings.home_dir` — for documents outside the project
///   but still inside the framework;
/// * the bare filename — so a path from somewhere else entirely still tells
///   the client what the file is without disclosing where it lives.
///
/// Always returns POSIX separators so a path rendered on Windows and one
/// rendered on macOS compare equal.  An empty or missing `path` is returned
/// unchanged.
pub fn relative_document_path(path: Option<&str>, project_dir: Option<&str>) -> Option<String> {
    todo!()
}

/// Processing state of a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentStatus {
    #[default]
    Pending,
    Converting,
    Indexing,
    Failed,
    Indexed,
    Deleted,
    Processed,
}

impl DocumentStatus {
    /// The wire value (`"pending"`, `"indexed"`, ...), i.e. Python's `.value`.
    pub fn as_str(&self) -> &'static str {
        match self {
            DocumentStatus::Pending => "pending",
            DocumentStatus::Converting => "converting",
            DocumentStatus::Indexing => "indexing",
            DocumentStatus::Failed => "failed",
            DocumentStatus::Indexed => "indexed",
            DocumentStatus::Deleted => "deleted",
            DocumentStatus::Processed => "processed",
        }
    }
}

impl std::fmt::Display for DocumentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A document registered to a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Document {
    pub id: String,
    pub path: String,
    /// Original filename — set once, never overwritten by the indexer.
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub status: DocumentStatus,
}

impl Document {
    /// Return a serialization-ready view of this document.
    ///
    /// `path` is emitted **relative**, never absolute.  An absolute path pins
    /// a client to one machine's layout and discloses the real home directory
    /// to every API consumer.
    pub fn to_info(&self, project_dir: Option<&str>) -> DocumentInfo {
        todo!()
    }
}

/// Serialization-ready view of a Document, suitable for API responses and SSE
/// events.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentInfo {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default = "default_document_info_status")]
    pub status: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub filename: Option<String>,
}

fn default_document_info_status() -> String {
    "pending".to_string()
}

impl Default for DocumentInfo {
    fn default() -> Self {
        Self {
            id: None,
            path: None,
            status: default_document_info_status(),
            summary: None,
            filename: None,
        }
    }
}

/// A skill (SKILL.md) available to a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Skill {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub content: String,
    #[serde(default)]
    pub system: bool,
}
